package nn;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import cn.fox.biomedical.Dictionary;
import cn.fox.biomedical.Sider;
import cn.fox.machine_learning.BrownCluster;
import cn.fox.nlp.EnglishPos;
import cn.fox.nlp.Punctuation;
import cn.fox.nlp.Word2Vec;
import cn.fox.stanford.Tokenizer;
import cn.fox.utils.CharCode;
import cn.fox.utils.Evaluater;
import cn.fox.utils.ObjectSerializer;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;
import drug_side_effect_utils.Tool;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.PropertiesUtils;
import gnu.trove.TIntArrayList;
import gnu.trove.TIntIntHashMap;
import gnu.trove.TObjectIntHashMap;
import sparse_pipeline.ClassifierEntity;
import sparse_pipeline.Prediction;
import sparse_pipeline.Util;
import utils.ADESentence;
import utils.Abstract;

public class NNADE extends Father implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6814631308712316743L;
	public NNSimple nn;
	public Parameters parameters;
	// dictionary
	public List<String> knownWords;
	public List<String> knownPos;
	public List<String> knownPreSuffix;
	public List<String> knownBrown;
	public List<String> knownSynSet;
	public List<String> knownHyper;
	public List<String> knownDict;
	public List<String> knownEntityType;
	public List<String> knownGlobalFeature;
	
	// key-word, value-word ID (the row id of the embedding matrix)
	public TObjectIntHashMap<String> wordIDs;
	public TObjectIntHashMap<String> posIDs;
	public TObjectIntHashMap<String> presuffixIDs;
	public TObjectIntHashMap<String> brownIDs;
	public TObjectIntHashMap<String> synsetIDs;
	public TObjectIntHashMap<String> hyperIDs;
	public TObjectIntHashMap<String> dictIDs;
	public TObjectIntHashMap<String> entitytypeIDs;
	public TObjectIntHashMap<String> globalFeatureIDs;
	
	// only used when loading external embeddings
	public TObjectIntHashMap<String> embedID;
	public double[][] embeddings;

	// the embedding matrix, embedding numbers x embeddingSize
	public double[][] E; // word embedding can use pre-trained
	public double[][] eg2E;
	

	
	// Store the high-frequency token-position
	public TIntArrayList preComputed;
	
	public boolean debug;
	

	public NNADE(Parameters parameters) {
		
		this.parameters = parameters;
	}

	public static void main(String[] args) throws Exception {
		FileInputStream fis = new FileInputStream(args[0]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
				
		Parameters parameters = new Parameters(properties);
		parameters.printParameters();
		
				
		File fAbstractDir = new File(PropertiesUtils.getString(properties, "corpusDir", ""));
		File groupFile = new File(PropertiesUtils.getString(properties, "groupFile", ""));
		String modelFile = PropertiesUtils.getString(properties, "modelFile", "");
		String embedFile = PropertiesUtils.getString(properties, "embedFile", "");
		String wordListFile = PropertiesUtils.getString(properties, "wordListFile", "");
		String secondEmbFile = PropertiesUtils.getString(properties, "secondEmbFile", "");
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(groupFile), "utf-8"));
		String thisLine = null;
		List<Set<String>> groups = new ArrayList<>();
		Set<String> oneGroup = new HashSet<>();
		while ((thisLine = br.readLine()) != null) {
			if(thisLine.isEmpty()) {
				groups.add(oneGroup);
				oneGroup = new HashSet<>();
			} else
				oneGroup.add(thisLine.trim());
		}
		groups.add(oneGroup);
		br.close();
		
		System.out.println("Abstract Dir "+ fAbstractDir.getAbsolutePath());
		System.out.println("Group File "+ groupFile.getAbsolutePath());
		System.out.println("Embedding File " + embedFile);
		
		if(!parameters.bEmbeddingFineTune && (embedFile == null || embedFile.isEmpty())) {
			throw new Exception();
		}
		
		Word2Vec w2v = new Word2Vec();
		if(embedFile != null && !embedFile.isEmpty()) {
			long startTime = System.currentTimeMillis();
			w2v.loadModel(embedFile, true);
			System.out.println("Load main pretrained embeddings using " + ((System.currentTimeMillis()-startTime) / 1000.0)+"s");
		}
		
		
		// load relevant tools
		Tool tool = new Tool();
		tool.tokenizer = new Tokenizer(true, ' ');	
		tool.tagger = new MaxentTagger(PropertiesUtils.getString(properties, "pos_tagger", ""));
		tool.morphology = new Morphology();
		BrownCluster brown = new BrownCluster(PropertiesUtils.getString(properties, "brown_cluster_path", ""), 100);
		IDictionary dict = new edu.mit.jwi.Dictionary(new URL("file", null, PropertiesUtils.getString(properties, "wordnet_dict", "")));
		dict.open();
		tool.brownCluster = brown;
		tool.dict = dict;
		
		Dictionary jochem = new Dictionary(PropertiesUtils.getString(properties, "jochem_dict", ""), 6);
		Dictionary ctdchem = new Dictionary(PropertiesUtils.getString(properties, "ctdchem_dict", ""), 6);
		Dictionary chemElem = new Dictionary(PropertiesUtils.getString(properties, "chemical_element_abbr", ""), 1);
		Dictionary drugbank = new Dictionary(PropertiesUtils.getString(properties, "drug_dict", ""), 6);
		tool.jochem = jochem;
		tool.ctdchem = ctdchem;
		tool.chemElem = chemElem;
		tool.drugbank = drugbank;
		
		Dictionary ctdmedic = new Dictionary(PropertiesUtils.getString(properties, "ctdmedic_dict", ""), 6);
		Dictionary humando = new Dictionary(PropertiesUtils.getString(properties, "disease_dict", ""), 6);
		tool.ctdmedic = ctdmedic;
		tool.humando = humando;
		
		Sider sider = new Sider(PropertiesUtils.getString(properties, "sider_dict", ""));
		tool.sider = sider;
		
		List<BestPerformance> bestAll = new ArrayList<>();
		for(int i=0;i<groups.size();i++) {
			Set<String> group = groups.get(i);
			Set<String> groupDev = groups.get((i+1)%groups.size());
			
			List<Abstract> devAb = new ArrayList<>();
			List<Abstract> trainAb = new ArrayList<>();
			List<Abstract> testAb = new ArrayList<>();
			for(File abstractFile:fAbstractDir.listFiles()) {
				Abstract ab = (Abstract)ObjectSerializer.readObjectFromFile(abstractFile.getAbsolutePath());
				if(group.contains(ab.id)) {
					// test abstract
					testAb.add(ab);
				} else if(groupDev.contains(ab.id)) {
					// dev abstract
					devAb.add(ab);
				} else {
					// train abstract
					trainAb.add(ab);
				}
				

			}
			
			NNADE nnade = new NNADE(parameters);
			nnade.debug = Boolean.parseBoolean(args[1]);
			//nnade.brownCluster = brown;
			//nnade.wordnet = dict;
			// save model for each group
			System.out.println(Parameters.SEPARATOR+" group "+i);
			BestPerformance best = nnade.trainAndTest(trainAb, devAb, testAb,modelFile+i, embedFile, 
					tool, wordListFile, w2v);
			bestAll.add(best);
			
			break;
		}
		
		// dev
		double pDev_Entity = 0;
		double rDev_Entity = 0;
		double f1Dev_Entity = 0;
		double pDev_Relation = 0;
		double rDev_Relation = 0;
		double f1Dev_Relation = 0;
		for(BestPerformance best:bestAll) {
			pDev_Entity += best.dev_pEntity/bestAll.size(); 
			rDev_Entity += best.dev_rEntity/bestAll.size();
			pDev_Relation += best.dev_pRelation/bestAll.size();
			rDev_Relation += best.dev_rRelation/bestAll.size();
		}
		f1Dev_Entity = Evaluater.getFMeasure(pDev_Entity, rDev_Entity, 1);
		f1Dev_Relation = Evaluater.getFMeasure(pDev_Relation, rDev_Relation, 1);
		
		System.out.println("dev entity precision\t"+pDev_Entity);
        System.out.println("dev entity recall\t"+rDev_Entity);
        System.out.println("dev entity f1\t"+f1Dev_Entity);
        System.out.println("dev relation precision\t"+pDev_Relation);
        System.out.println("dev relation recall\t"+rDev_Relation);
        System.out.println("dev relation f1\t"+f1Dev_Relation);
        
        
        // test
        double pTest_Entity = 0;
		double rTest_Entity = 0;
		double f1Test_Entity = 0;
		double pTest_Relation = 0;
		double rTest_Relation = 0;
		double f1Test_Relation = 0;
		for(BestPerformance best:bestAll) {
			pTest_Entity += best.test_pEntity/bestAll.size(); 
			rTest_Entity += best.test_rEntity/bestAll.size();
			pTest_Relation += best.test_pRelation/bestAll.size();
			rTest_Relation += best.test_rRelation/bestAll.size();
		}
		f1Test_Entity = Evaluater.getFMeasure(pTest_Entity, rTest_Entity, 1);
		f1Test_Relation = Evaluater.getFMeasure(pTest_Relation, rTest_Relation, 1);
		
		System.out.println("test entity precision\t"+pTest_Entity);
        System.out.println("test entity recall\t"+rTest_Entity);
        System.out.println("test entity f1\t"+f1Test_Entity);
        System.out.println("test relation precision\t"+pTest_Relation);
        System.out.println("test relation recall\t"+rTest_Relation);
        System.out.println("test relation f1\t"+f1Test_Relation);
	}
	
	public BestPerformance trainAndTest(List<Abstract> trainAbs, List<Abstract> devAbs, 
			List<Abstract> testAbs, String modelFile, 
			String embedFile, Tool tool, String wordListFile, Word2Vec w2v) 
		throws Exception {
		// generate alphabet
		List<String> word = new ArrayList<>();
		List<String> pos = new ArrayList<>();	
		List<String> presuffix = new ArrayList<>();
		List<String> brown = new ArrayList<>();
		List<String> synset = new ArrayList<>();
		List<String> hyper = new ArrayList<>();
		List<String> dict = new ArrayList<>();
		List<String> entityType = new ArrayList<>();
		for(int i=1;i<=parameters.wordCutOff+1;i++) {
			entityType.add("Disease");
			entityType.add("Chemical");
		}
		
		List<String> globalFeatures = createGlobalFeatureAlphabet(trainAbs, tool);
		
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				
				for(CoreLabel token:tokens) {
					
					word.add(wordPreprocess(token, parameters));
					pos.add(token.tag());
					presuffix.add(getPrefix(token));
					presuffix.add(getSuffix(token));
					brown.add(getBrown(token, tool));
					dict.add(getDict(token, tool));

					
					String id = getSynset(token, tool);
					if(id!=null)
						synset.add(id);
					
					String hyperID = getHyper(token, tool);
					if(hyperID!=null)
						hyper.add(hyperID);
				}
				
				
				for(Entity entity:sentence.entities) {
					String id = getSynset(entity.text, tool);
					if(id!=null)
						synset.add(id);
					
					String hyperID = getHyper(entity.text, tool);
					if(hyperID!=null)
						hyper.add(hyperID);
					
				}
			}
		}
		
		if(!parameters.bEmbeddingFineTune) {
			// add the alphabet of the test set
			for(Abstract ab:testAbs) { 
				for(ADESentence sentence:ab.sentences) {
					// for each sentence
					List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
					
					for(CoreLabel token:tokens) {
						
						word.add(wordPreprocess(token, parameters));
						pos.add(token.tag());
						presuffix.add(getPrefix(token));
						presuffix.add(getSuffix(token));
						brown.add(getBrown(token, tool));
						dict.add(getDict(token, tool));
						
						String id = getSynset(token, tool);
						if(id!=null)
							synset.add(id);
						
						String hyperID = getHyper(token, tool);
						if(hyperID!=null)
							hyper.add(hyperID);
					}
					
					
					for(Entity entity:sentence.entities) {
						String id = getSynset(entity.text, tool);
						if(id!=null)
							synset.add(id);
						
						String hyperID = getHyper(entity.text, tool);
						if(hyperID!=null)
							hyper.add(hyperID);
						
					}
				}
			}
			
		}
		
		if(parameters.bEmbeddingFineTune && parameters.wordCutOff<=1)
			throw new Exception();
			
		knownWords = Util.generateDict(word, parameters.wordCutOff);
	    knownWords.add(0, Parameters.UNKNOWN);
	    knownWords.add(1, Parameters.PADDING);
	    knownPos = Util.generateDict(pos, parameters.wordCutOff);
	    knownPos.add(0, Parameters.UNKNOWN);
	    knownPos.add(1, Parameters.PADDING);
	    knownPreSuffix = Util.generateDict(presuffix, parameters.wordCutOff);
	    knownPreSuffix.add(0, Parameters.UNKNOWN);
	    knownPreSuffix.add(1, Parameters.PADDING);
	    knownBrown = Util.generateDict(brown, parameters.wordCutOff);
	    knownBrown.add(0, Parameters.UNKNOWN);
	    knownBrown.add(1, Parameters.PADDING);
	    knownSynSet = Util.generateDict(synset, parameters.wordCutOff);
	    knownSynSet.add(0, Parameters.UNKNOWN);
	    knownSynSet.add(1, Parameters.PADDING);
	    knownHyper = Util.generateDict(hyper, parameters.wordCutOff);
	    knownHyper.add(0, Parameters.UNKNOWN);
	    knownHyper.add(1, Parameters.PADDING);
	    knownDict = Util.generateDict(dict, parameters.wordCutOff);
	    knownDict.add(Parameters.UNKNOWN);
	    knownDict.add(Parameters.PADDING);
	    knownEntityType = Util.generateDict(entityType, parameters.wordCutOff);
	    knownEntityType.add(Parameters.UNKNOWN);
	    
	    knownGlobalFeature = Util.generateDict(globalFeatures, parameters.wordCutOff);
	    knownGlobalFeature.add(Parameters.UNKNOWN);
	    		
	    
	    // Generate word id which can be used in the embedding matrix
	    wordIDs = new TObjectIntHashMap<String>();
	    posIDs = new TObjectIntHashMap<>();
	    presuffixIDs = new TObjectIntHashMap<>();
	    brownIDs = new TObjectIntHashMap<>();
	    synsetIDs = new TObjectIntHashMap<>();
	    hyperIDs = new TObjectIntHashMap<>();
	    dictIDs = new TObjectIntHashMap<>();
	    entitytypeIDs = new TObjectIntHashMap<>();
	    globalFeatureIDs = new TObjectIntHashMap<>();
	    int m = 0;
	    for (String temp : knownWords)
	      wordIDs.put(temp, (m++));
	    for (String temp : knownPos)
	        posIDs.put(temp, (m++));
	    for (String temp : knownPreSuffix)
	    	presuffixIDs.put(temp, (m++));
	    for (String temp : knownBrown)
	    	brownIDs.put(temp, (m++));
	    for(String temp:knownSynSet)
	    	synsetIDs.put(temp, (m++));
	    for(String temp:knownHyper)
	    	hyperIDs.put(temp, (m++));
	    for(String temp:knownDict)
	    	dictIDs.put(temp, (m++));
	    for(String temp:knownEntityType)
	    	entitytypeIDs.put(temp, (m++));
	    for(String temp:knownGlobalFeature)
	    	globalFeatureIDs.put(temp, (m++));

	    System.out.println("#Word: " + knownWords.size());
	    System.out.println("#POS: " + knownPos.size());
	    System.out.println("#PreSuffix: " + knownPreSuffix.size());
	    System.out.println("#Brown: " + knownBrown.size());
	    System.out.println("#Synset: " + knownSynSet.size());
	    System.out.println("#Hyper: " + knownHyper.size());
	    System.out.println("#Dict: " + knownDict.size());
	    System.out.println("#Entity type: "+knownEntityType.size());
	    System.out.println("#Global feature: "+knownGlobalFeature.size());
	    
	    E = new double[m][parameters.embeddingSize];
		eg2E = new double[E.length][E[0].length];
		
	    if(parameters.bEmbeddingFineTune) {
	    	if(embedFile!=null && !embedFile.isEmpty()) {
	    		TIntArrayList uninitialIds = new TIntArrayList();
	    		int unknownID = -1;
				double sum[] = new double[parameters.embeddingSize];
				int count = 0;
				for (int i = 0; i < knownWords.size(); ++i) {
					if(knownWords.get(i).equals(Parameters.UNKNOWN)) {
						unknownID = wordIDs.get(knownWords.get(i));
						continue;
					}
					
					String str = knownWords.get(i);
				      int id = wordIDs.get(str);
				      if (w2v.wordMap.containsKey(str))  {
				    	  for(int j=0;j<E[0].length;j++) {
				    		  E[id][j] = w2v.wordMap.get(str)[j];
				    		  sum[j] += E[id][j];
				    	  }
				    	  count++;
				      } else {
				    	  uninitialIds.add(id);
				      }
				}
				// unkown is the average of all words
				for (int idx = 0; idx < parameters.embeddingSize; idx++) {
				   E[unknownID][idx] = sum[idx] / count;
				}
				// word not in pre-trained embedding will use the unknown embedding
				for(int i=0;i<uninitialIds.size();i++) {
					int id = uninitialIds.get(i);
					for (int j = 0; j < parameters.embeddingSize; j++) {
						E[id][j] = E[unknownID][j];
					}
				}
	    	} else {
	    		Random random = new Random(System.currentTimeMillis());
	    		int unknownID = -1;
				double sum[] = new double[parameters.embeddingSize];
				int count = 0;
	    		for (int i = 0; i < knownWords.size(); ++i) {
	    			if(knownWords.get(i).equals(Parameters.UNKNOWN)) {
						unknownID = wordIDs.get(knownWords.get(i));
						continue;
					}
	    			
	    			
	    			String str = knownWords.get(i);
				    int id = wordIDs.get(str);
	    			double norm = 0;
					for(int j=0;j<E[0].length;j++) {
				    	E[id][j] = random.nextDouble() * parameters.initRange * 2 - parameters.initRange;
				    	norm += E[id][j]*E[id][j];
				    	
				    }
					norm = Math.sqrt(norm);
					for(int j=0;j<E[0].length;j++) {
						E[id][j] = E[id][j]/norm;
						sum[j] += E[id][j];
					}
				    count++; 
				}
	    		for (int idx = 0; idx < parameters.embeddingSize; idx++) {
	 			   E[unknownID][idx] = sum[idx] / count;
	 			}
	    		
	    		
	    	}
	    } else {
	    	if(embedFile==null && embedFile.isEmpty())
	    		throw new Exception ();
	    	
	    	TIntArrayList uninitialIds = new TIntArrayList();
			int unknownID = -1;
			double sum[] = new double[parameters.embeddingSize];
			int count = 0;
			for (int i = 0; i < knownWords.size(); ++i) {
				if(knownWords.get(i).equals(Parameters.UNKNOWN)) {
					unknownID = wordIDs.get(knownWords.get(i));
					continue;
				}
				
			      String str = knownWords.get(i);
			      int id = wordIDs.get(str);
			      if (w2v.wordMap.containsKey(str))  {
			    	  for(int j=0;j<E[0].length;j++) {
			    		  E[id][j] = w2v.wordMap.get(str)[j];
			    		  sum[j] += E[id][j];
			    	  }
			    	  count++;
			      } else {
			    	  uninitialIds.add(id);
			      }
			}
			// unkown is the average of all words
			for (int idx = 0; idx < parameters.embeddingSize; idx++) {
			   E[unknownID][idx] = sum[idx] / count;
			}
			// word not in pre-trained embedding will use the unknown embedding
			for(int i=0;i<uninitialIds.size();i++) {
				int id = uninitialIds.get(i);
				for (int j = 0; j < parameters.embeddingSize; j++) {
					E[id][j] = E[unknownID][j];
				}
			}
	    }
	    
		
		// non-word embedding can only be initialized randomly
		randomInitialEmbedding(knownPos, posIDs, E);
		randomInitialEmbedding(knownPreSuffix, presuffixIDs, E);
		randomInitialEmbedding(knownBrown, brownIDs, E);
		randomInitialEmbedding(knownSynSet, synsetIDs, E);
		randomInitialEmbedding(knownHyper, hyperIDs, E);
		randomInitialEmbedding(knownDict, dictIDs, E);
		randomInitialEmbedding(knownEntityType, entitytypeIDs, E);
		randomInitialEmbedding(knownGlobalFeature, globalFeatureIDs, E);

		// generate training examples
		Counter<Integer> tokPosCount = new IntCounter<>();
		List<Example> examples = generateTrainExamples(trainAbs, tokPosCount, tool);
		System.out.println("non-composite feature number: "+examples.get(0).featureIdx.size());
		// initialize preComputed
		preComputed = new TIntArrayList();
	    List<Integer> sortedTokens = Counters.toSortedList(tokPosCount, false);
	    List<Integer> sublist = sortedTokens.subList(0, Math.min(parameters.numPreComputed, sortedTokens.size()));
	    for(int tokPos : sublist) {
	    	preComputed.add(tokPos);
	    }
		
		// new a NN and initialize its weight
	    nn  = new NNSimple(parameters, this, preComputed, examples.get(0));
		nn.debug = debug;
		
		// train iteration
		long startTime = System.currentTimeMillis();
		BestPerformance best = new BestPerformance();
		
		int inputSize = examples.size();
		int batchBlock = inputSize / parameters.batchSize;
		if (inputSize % parameters.batchSize != 0)
			batchBlock++;
		
		  TIntArrayList indexes = new TIntArrayList();
		  for (int i = 0; i < inputSize; ++i)
		    indexes.add(i);
		  
		  List<Example> subExamples = new ArrayList<>();
		
		for (int iter = 0; iter < parameters.maxIter; ++iter) {
			if(debug)
				System.out.println("##### Iteration " + iter);
			
			for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * parameters.batchSize;
				int end_pos = (updateIter + 1) * parameters.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.add(examples.get(indexes.get(idy)));
				}
				
				GradientKeeper1 keeper = nn.process(subExamples, null);
				nn.updateWeights(keeper);
			}
			
			if(debug)
				System.out.println("Elapsed Time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");
			
			if (iter>0 && iter % parameters.evalPerIter == 0) {
				evaluate(tool, devAbs, testAbs, modelFile, best, false);
			}			
		}
		
		evaluate(tool, devAbs, testAbs, modelFile, best, true);
		
		return best;
	}
	
	// random, norm and average unknown
	public void randomInitialEmbedding(List<String> known, TObjectIntHashMap<String> IDs, 
			double[][] emb) {
		Random random = new Random(System.currentTimeMillis());
		int unknownID = -1;
		double sum[] = new double[parameters.embeddingSize];
		int count = 0;
		for (int i = 0; i < known.size(); ++i) {
			if(known.get(i).equals(Parameters.UNKNOWN)) {
				unknownID = IDs.get(known.get(i));
				continue;
			}
			
			
			String str = known.get(i);
		    int id = IDs.get(str);
			double norm = 0;
			for(int j=0;j<emb[0].length;j++) {
		    	emb[id][j] = random.nextDouble() * parameters.initRange * 2 - parameters.initRange;
		    	norm += emb[id][j]*emb[id][j];
		    	
		    }
			norm = Math.sqrt(norm);
			for(int j=0;j<emb[0].length;j++) {
				emb[id][j] = emb[id][j]/norm;
				sum[j] += emb[id][j];
			}
		    count++; 
		}
		for (int idx = 0; idx < parameters.embeddingSize; idx++) {
		   emb[unknownID][idx] = sum[idx] / count;
		}
	}
	
	// Evaluate with the test set, and if the f1 is higher than bestF1, save the model
	public void evaluate(Tool tool, List<Abstract> devAbs, 
			List<Abstract> testAbs, String modelFile, BestPerformance best, boolean printerror)
			throws Exception {
		// Redo precomputation with updated weights. This is only
        // necessary because we're updating weights -- for normal
        // prediction, we just do this once 
        nn.preCompute();

        DecodeStatistic stat = new DecodeStatistic();
        for(Abstract devAb:devAbs) {
        	if(printerror) {
	        	System.out.println(Parameters.SEPARATOR);
	        	System.out.println("Document: "+devAb.id);
        	}
        	
        	for(ADESentence gold:devAb.sentences) {
        		List<CoreLabel> tokens = prepareNLPInfo(tool, gold);
        		ADESentence predicted = null;
        		predicted = decode(tokens, tool);
        		
        		stat.ctPredictEntity += predicted.entities.size();
        		stat.ctTrueEntity += gold.entities.size();
        		for(Entity preEntity:predicted.entities) {
        			if(gold.entities.contains(preEntity))
        				stat.ctCorrectEntity++;
    			}
        		
        		
        		stat.ctPredictRelation += predicted.relaitons.size();
        		stat.ctTrueRelation += gold.relaitons.size();
        		for(RelationEntity preRelation:predicted.relaitons) {
        			if(gold.relaitons.contains(preRelation))
        				stat.ctCorrectRelation++;
        		}
        		
        		if(printerror) {
	        		System.out.println("Gold Entity: ");
	        		for(Entity entity:gold.entities)
	        			System.out.println(entity);
	        		System.out.println("Predict Entity: ");
	        		for(Entity entity:predicted.entities)
	        			System.out.println(entity);
	        		System.out.println("Gold relation: ");
	        		for(RelationEntity re:gold.relaitons)
	        			System.out.println(re);
	        		System.out.println("Predict relation: ");
	        		for(RelationEntity re:predicted.relaitons)
	        			System.out.println(re);
        		}

        	}
        	
        	if(printerror) {
        		System.out.println(Parameters.SEPARATOR);
        	}
        }
        
        if(printerror) {
	        System.out.println(Parameters.SEPARATOR);
	        System.out.println(stat.getEntityTP());
	        System.out.println(stat.getEntityFP());
	        System.out.println(stat.getEntityFN());
	        System.out.println(stat.getRelationTP());
	        System.out.println(stat.getRelationFP());
	        System.out.println(stat.getRelationFN());
	        System.out.println(Parameters.SEPARATOR);
	        return;
        }
        
        System.out.println(Parameters.SEPARATOR);
        double dev_pEntity = stat.getEntityPrecision();
        System.out.println("dev entity precision\t"+dev_pEntity);
        double dev_rEntity = stat.getEntityRecall();
        System.out.println("dev entity recall\t"+dev_rEntity);
        double dev_f1Entity = stat.getEntityF1();
        System.out.println("dev entity f1\t"+dev_f1Entity);
        double dev_pRelation = stat.getRelationPrecision();
        System.out.println("dev relation precision\t"+dev_pRelation);
        double dev_rRelation = stat.getRelationRecall();
        System.out.println("dev relation recall\t"+dev_rRelation);
        double dev_f1Relation = stat.getRelationF1();
        System.out.println("dev relation f1\t"+dev_f1Relation);

        	
        if ((dev_f1Relation > best.dev_f1Relation) || (dev_f1Relation==best.dev_f1Relation && dev_f1Entity>best.dev_f1Entity)) {
        //if ((f1Entity > best.f1Entity)) {
          System.out.printf("Current Exceeds the best! Saving model file %s\n", modelFile);
          best.dev_pEntity = dev_pEntity;
          best.dev_rEntity = dev_rEntity;
          best.dev_f1Entity = dev_f1Entity;
          best.dev_pRelation = dev_pRelation;
          best.dev_rRelation = dev_rRelation;
          best.dev_f1Relation = dev_f1Relation;
          //ObjectSerializer.writeObjectToFile(this, modelFile);
          
          // the current outperforms the prior on dev, so we evaluate on test and record the performance
          DecodeStatistic stat2 = new DecodeStatistic();
          for(Abstract testAb:testAbs) {
          	for(ADESentence gold:testAb.sentences) {
          		List<CoreLabel> tokens = prepareNLPInfo(tool, gold);
          		ADESentence predicted = null;
          		predicted = decode(tokens, tool);
          		
          		stat2.ctPredictEntity += predicted.entities.size();
          		stat2.ctTrueEntity += gold.entities.size();
          		for(Entity preEntity:predicted.entities) {
          			if(gold.entities.contains(preEntity))
          				stat2.ctCorrectEntity++;
      			}
          		
          		stat2.ctPredictRelation += predicted.relaitons.size();
        		stat2.ctTrueRelation += gold.relaitons.size();
        		for(RelationEntity preRelation:predicted.relaitons) {
        			if(gold.relaitons.contains(preRelation))
        				stat2.ctCorrectRelation++;
        		}
          		
          	}
          }
          
          
          double test_pEntity = stat2.getEntityPrecision();
          System.out.println("test entity precision\t"+test_pEntity);
          double test_rEntity = stat2.getEntityRecall();
          System.out.println("test entity recall\t"+test_rEntity);
          double test_f1Entity = stat2.getEntityF1();
          System.out.println("test entity f1\t"+test_f1Entity);
          double test_pRelation = stat2.getRelationPrecision();
          System.out.println("test relation precision\t"+test_pRelation);
          double test_rRelation = stat2.getRelationRecall();
          System.out.println("test relation recall\t"+test_rRelation);
          double test_f1Relation = stat2.getRelationF1();
          System.out.println("test relation f1\t"+test_f1Relation);
          System.out.println(Parameters.SEPARATOR);
          // update the best test performance
          best.test_pEntity = test_pEntity;
          best.test_rEntity = test_rEntity;
          best.test_f1Entity = test_f1Entity;
          best.test_pRelation = test_pRelation;
          best.test_rRelation = test_rRelation;
          best.test_f1Relation = test_f1Relation;
        } else {
        	System.out.println(Parameters.SEPARATOR);
        }
        
	}
	
	public List<Example> generateTrainExamples(List<Abstract> trainAbs, Counter<Integer> tokPosCount, Tool tool)
			throws Exception  {
		
		List<Example> ret = new ArrayList<>();
				
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				// resort the entities in the sentence
				List<Entity> entities = Util.resortEntity(sentence);
				// fill 'start' and 'end' of the entities
				//Util.fillEntity(entities, tokens);
				
				Prediction prediction = new Prediction();
				// for each token, we generate an entity example
				for(int idx=0;idx<tokens.size();idx++) {
					// prepare the input for NN
					Example example = getExampleFeatures(tokens, idx, false, null, null, tool, prediction);
					double[] goldLabel = {0,0,0,0,0,0};
					CoreLabel token = tokens.get(idx);
					int index = Util.isInsideAGoldEntityAndReturnIt(token.beginPosition(), token.endPosition(), entities);
					int transition = -1;
					if(index == -1) {
						// other
						goldLabel[0] = 1;
						transition = 0;
					} else {
						Entity gold = entities.get(index);
						if(Util.isFirstWordOfEntity(gold, token)) {
							if(gold.type.equals(Parameters.CHEMICAL)) {
								// new chemical
								goldLabel[1] = 1;
								transition = 1;
							} else {
								// new disease
								goldLabel[2] = 1;
								transition = 2;
							}
						} else {
							// append
							goldLabel[3] = 1;
							transition = 3;
						}
					}
					prediction.addLabel(transition, -1);
					example.label = goldLabel;
					ret.add(example);
					
					for(int j=0; j<example.featureIdx.size(); j++)
						if(example.featureIdx.get(j) != -1)
							tokPosCount.incrementCount(example.featureIdx.get(j)*example.featureIdx.size()+j);
					
					// generate entities based on the latest label
					int curTran = prediction.labels.get(prediction.labels.size()-1);
					if(curTran==1) { // new chemical
						CoreLabel current = tokens.get(idx);
						  Entity chem = new Entity(null, Parameters.CHEMICAL, current.beginPosition(), 
								  current.word(), null);
						  chem.start = idx;
						  chem.end = idx;
						  prediction.entities.add(chem);
					} else if(curTran==2) {// new disease
						CoreLabel current = tokens.get(idx);
						  Entity disease = new Entity(null, Parameters.DISEASE, current.beginPosition(), 
								  current.word(), null);
						  disease.start = idx;
						  disease.end = idx;
						  prediction.entities.add(disease);
					} else if(curTran==3 && checkWrongState(prediction)) { // append the current entity
						Entity old = prediction.entities.get(prediction.entities.size()-1);
						CoreLabel current = tokens.get(idx);
						int whitespaceToAdd = current.beginPosition()-(old.offset+old.text.length());
						for(int j=0;j<whitespaceToAdd;j++)
							old.text += " ";
						old.text += current.word();
						old.end = idx;
					}
					
					// begin to predict relations
					int lastTran = prediction.labels.size()>=2 ? prediction.labels.get(prediction.labels.size()-2) : -1;
					// judge whether to generate relations
					if((lastTran==1 && curTran==1) || (lastTran==1 && curTran==2) || (lastTran==1 && curTran==0)
							|| (lastTran==2 && curTran==0) || (lastTran==2 && curTran==1) || (lastTran==2 && curTran==2)
							|| (lastTran==3 && curTran==0) || (lastTran==3 && curTran==1) || (lastTran==3 && curTran==2)
					) { 
						if((lastTran==3 && checkWrongState(prediction)) || lastTran==1 || lastTran==2) {
							// if curTran 1 or 2, the last entities should not be considered
							int latterIdx = (curTran==1 || curTran==2) ? prediction.entities.size()-2:prediction.entities.size()-1;
							Entity latter = prediction.entities.get(latterIdx);
							for(int j=0;j<latterIdx;j++) {
								Entity former = prediction.entities.get(j);
								if(latter.type.equals(former.type))
									continue;
								Example relationExample = getExampleFeatures(tokens, -1, true, former, latter, tool, prediction);
								double[] relationGoldLabel = {0,0,0,0,0,0};	
								transition = -1;
								RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
								if(sentence.relaitons.contains(tempRelation)) {
									// connect
									relationGoldLabel[5] = 1;
									transition = 5;
								} else {
									// not connect
									relationGoldLabel[4] = 1;
									transition = 4;
								}
								relationExample.label = relationGoldLabel;
								ret.add(relationExample);
								
								for(int k=0; k<example.featureIdx.size(); k++)
									if(example.featureIdx.get(k) != -1)
										tokPosCount.incrementCount(example.featureIdx.get(k)*example.featureIdx.size()+k);
								
								prediction.addLabel(transition,-1);

					            // generate relations based on the latest label
					            curTran = prediction.labels.get(prediction.labels.size()-1);
					        	if(curTran == 5) { // connect
									RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
									prediction.relations.add(relationEntity);
					        	}

							}
						}
						
					}
										
				}
				
				
				// when at the end of sentence, judge relation ignoring lastTran
				int curTran = prediction.labels.get(prediction.labels.size()-1);
				if((curTran==3 && checkWrongState(prediction)) || curTran==1 || curTran==2) {
					int latterIdx = prediction.entities.size()-1;
					Entity latter = prediction.entities.get(latterIdx);

					for(int j=0;j<latterIdx;j++) {
						Entity former = prediction.entities.get(j);
						if(latter.type.equals(former.type))
							continue;
						Example relationExample = getExampleFeatures(tokens, -1, true, former, latter, tool, prediction);
						double[] relationGoldLabel = {0,0,0,0,0,0};	
						int transition = -1;
						RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
						if(sentence.relaitons.contains(tempRelation)) {
							// connect
							relationGoldLabel[5] = 1;
							transition = 5;
						} else {
							// not connect
							relationGoldLabel[4] = 1;
							transition = 4;
						}
						relationExample.label = relationGoldLabel;
						ret.add(relationExample);
						
						for(int k=0; k<relationExample.featureIdx.size(); k++)
							if(relationExample.featureIdx.get(k) != -1)
								tokPosCount.incrementCount(relationExample.featureIdx.get(k)*relationExample.featureIdx.size()+k);
						
						prediction.addLabel(transition, -1);

			            // generate relations based on the latest label
			            curTran = prediction.labels.get(prediction.labels.size()-1);
			        	if(curTran == 5) { // connect
							RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
							prediction.relations.add(relationEntity);
			        	}

					}
					
				}
				
				
				
			}
		}
		
		return ret;
	
	}
	
	public List<String> createGlobalFeatureAlphabet(List<Abstract> trainAbs, Tool tool)
			throws Exception  {
		List<String> GlobalFeature = new ArrayList<>();
		
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				// resort the entities in the sentence
				List<Entity> entities = Util.resortEntity(sentence);
				// fill 'start' and 'end' of the entities
				//Util.fillEntity(entities, tokens);
				
				Prediction prediction = new Prediction();
				// for each token, we generate an entity example
				for(int idx=0;idx<tokens.size();idx++) {
					// prepare the input for NN
					List<String> features = getGlobalAlphabet(tokens, idx, false, null, null, tool, prediction);
					GlobalFeature.addAll(features);
					CoreLabel token = tokens.get(idx);
					int index = Util.isInsideAGoldEntityAndReturnIt(token.beginPosition(), token.endPosition(), entities);
					int transition = -1;
					if(index == -1) {
						// other
						transition = 0;
					} else {
						Entity gold = entities.get(index);
						if(Util.isFirstWordOfEntity(gold, token)) {
							if(gold.type.equals(Parameters.CHEMICAL)) {
								// new chemical
								transition = 1;
							} else {
								// new disease
								transition = 2;
							}
						} else {
							// append
							transition = 3;
						}
					}
					prediction.addLabel(transition, -1);

										
					// generate entities based on the latest label
					int curTran = prediction.labels.get(prediction.labels.size()-1);
					if(curTran==1) { // new chemical
						CoreLabel current = tokens.get(idx);
						  Entity chem = new Entity(null, Parameters.CHEMICAL, current.beginPosition(), 
								  current.word(), null);
						  chem.start = idx;
						  chem.end = idx;
						  prediction.entities.add(chem);
					} else if(curTran==2) {// new disease
						CoreLabel current = tokens.get(idx);
						  Entity disease = new Entity(null, Parameters.DISEASE, current.beginPosition(), 
								  current.word(), null);
						  disease.start = idx;
						  disease.end = idx;
						  prediction.entities.add(disease);
					} else if(curTran==3 && checkWrongState(prediction)) { // append the current entity
						Entity old = prediction.entities.get(prediction.entities.size()-1);
						CoreLabel current = tokens.get(idx);
						int whitespaceToAdd = current.beginPosition()-(old.offset+old.text.length());
						for(int j=0;j<whitespaceToAdd;j++)
							old.text += " ";
						old.text += current.word();
						old.end = idx;
					}
					
					// begin to predict relations
					int lastTran = prediction.labels.size()>=2 ? prediction.labels.get(prediction.labels.size()-2) : -1;
					// judge whether to generate relations
					if((lastTran==1 && curTran==1) || (lastTran==1 && curTran==2) || (lastTran==1 && curTran==0)
							|| (lastTran==2 && curTran==0) || (lastTran==2 && curTran==1) || (lastTran==2 && curTran==2)
							|| (lastTran==3 && curTran==0) || (lastTran==3 && curTran==1) || (lastTran==3 && curTran==2)
					) { 
						if((lastTran==3 && checkWrongState(prediction)) || lastTran==1 || lastTran==2) {
							// if curTran 1 or 2, the last entities should not be considered
							int latterIdx = (curTran==1 || curTran==2) ? prediction.entities.size()-2:prediction.entities.size()-1;
							Entity latter = prediction.entities.get(latterIdx);
							for(int j=0;j<latterIdx;j++) {
								Entity former = prediction.entities.get(j);
								if(latter.type.equals(former.type))
									continue;
								features = getGlobalAlphabet(tokens, -1, true, former, latter, tool, prediction);
								GlobalFeature.addAll(features);
								transition = -1;
								RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
								if(sentence.relaitons.contains(tempRelation)) {
									// connect
									transition = 5;
								} else {
									// not connect
									transition = 4;
								}

								
								prediction.addLabel(transition,-1);

					            // generate relations based on the latest label
					            curTran = prediction.labels.get(prediction.labels.size()-1);
					        	if(curTran == 5) { // connect
									RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
									prediction.relations.add(relationEntity);
					        	}

							}
						}
						
					}
										
				}
				
				
				// when at the end of sentence, judge relation ignoring lastTran
				int curTran = prediction.labels.get(prediction.labels.size()-1);
				if((curTran==3 && checkWrongState(prediction)) || curTran==1 || curTran==2) {
					int latterIdx = prediction.entities.size()-1;
					Entity latter = prediction.entities.get(latterIdx);

					for(int j=0;j<latterIdx;j++) {
						Entity former = prediction.entities.get(j);
						if(latter.type.equals(former.type))
							continue;
						List<String>featrues = getGlobalAlphabet(tokens, -1, true, former, latter, tool, prediction);
						GlobalFeature.addAll(featrues);
						int transition = -1;
						RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
						if(sentence.relaitons.contains(tempRelation)) {
							// connect
							transition = 5;
						} else {
							// not connect
							transition = 4;
						}

						
						
						prediction.addLabel(transition, -1);

			            // generate relations based on the latest label
			            curTran = prediction.labels.get(prediction.labels.size()-1);
			        	if(curTran == 5) { // connect
							RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
							prediction.relations.add(relationEntity);
			        	}

					}
					
				}
				
				
				
			}
		}
		
		return GlobalFeature;
	}

	public List<String> getGlobalAlphabet(List<CoreLabel> tokens, int idx, boolean bRelation,
			Entity former, Entity latter, Tool tool, Prediction predict) throws Exception {
		List<String> features = new ArrayList<>();
		
		if(!bRelation) {
			{ // is token in any ADE
				String current = tokens.get(idx).word().toLowerCase();
				boolean b = false; 
				int isIn = 0; // 0-not, 1-disease, 2-chemical
				  for(RelationEntity relation:predict.relations) {
					  for(int i=relation.getDisease().start;i<=relation.getDisease().end;i++) {
						  if(tokens.get(i).word().toLowerCase().equals(current)) {
							  isIn = 1;
							  b = true;
							  
							  break;
						  }
					  }
					  
					  if(isIn == 0) {
						  for(int i=relation.getChemical().start;i<=relation.getChemical().end;i++) {
							  if(tokens.get(i).word().toLowerCase().equals(current)) {
								  isIn = 2;
								  b = true;
								  
								  break;
							  }
						  }
					  }
					  
					  if(isIn != 0)
						  break;
						  
				  }
				  

				  features.add("GLOFEA2#"+isIn);	 
				
			}
			
			{ // is token's brown in any ADE
				String tokBrown = getBrown(tokens.get(idx), tool);
				if(tokBrown != null) {
					boolean b = false;
					int isIn = 0; // 0-not, 1-disease, 2-chemical
					for(RelationEntity relation:predict.relations) {
						
						  for(int i=relation.getDisease().start;i<=relation.getDisease().end;i++) {
							  String brownDisease = getBrown(tokens.get(i), tool);
							  if(brownDisease==null)
								  continue;
							  
							  if(tokBrown.equals(brownDisease)) {
								  isIn = 1;
								  b = true;
								  
								  break;
							  }
						  }
						  
						  if(isIn == 0) {
							  for(int i=relation.getChemical().start;i<=relation.getChemical().end;i++) {
								  String brownChem = getBrown(tokens.get(i), tool);
								  if(brownChem==null)
									  continue;
								  
								  if(tokBrown.equals(brownChem)) {
									  isIn = 2;
									  b = true;
									  
									  break;
								  }
							  }
						  }
						
						
						  if(isIn != 0)
							  break;
								
					}
					
					features.add("GLOFEA3#"+isIn);	
				} 
			}
				

			
			
			{ // is token's Synset in any ADE
				String toksynset = getSynset(tokens.get(idx), tool);
				if(toksynset != null) {
					boolean b = false;
					int isIn = 0; // 0-not, 1-disease, 2-chemical
					for(RelationEntity relation:predict.relations) {
						
						  for(int i=relation.getDisease().start;i<=relation.getDisease().end;i++) {
							  String synsetDisease = getSynset(tokens.get(i), tool);
							  if(synsetDisease==null)
								  continue;
							  
							  if(toksynset.equals(synsetDisease)) {
								  isIn = 1;
								  b = true;
								  
								  break;
							  }
						  }
						  
						  if(isIn == 0) {
							  for(int i=relation.getChemical().start;i<=relation.getChemical().end;i++) {
								  String synsetChem = getSynset(tokens.get(i), tool);
								  if(synsetChem==null)
									  continue;
								  
								  if(toksynset.equals(synsetChem)) {
									  isIn = 2;
									  b = true;
									  
									  break;
								  }
							  }
						  }
						
						
						  if(isIn != 0)
							  break;
								
					}
					
					features.add("GLOFEA4#"+isIn);	
				} 
				

			}
			
			
			{ // A -> B and tok
				Entity B = Util.getClosestCoorEntity(tokens, idx, predict);
				if(B != null) {
					
					int b = 0;
					  for(RelationEntity relation:predict.relations) {
						  if(relation.getDisease().text.toLowerCase().equals(B.text.toLowerCase())) {
							  b = 1;
							  break;
						  }
						  else if(relation.getChemical().text.toLowerCase().equals(B.text.toLowerCase()))  {
							  b = 2;
							  break;
						  }
					  }
					  
					  features.add("GLOFEA5#"+b);
				} 
			}
			
			
			{ 
				Entity previous = Util.getPreviousEntity(tokens, idx, predict);
				if(previous != null) {
					// whether the entity before tok is in relations
					int b = 0;
					for(RelationEntity relation:predict.relations) {
						  if(relation.getDisease().text.toLowerCase().equals(previous.text.toLowerCase())) {
							  b = 1;
							  break;
						  }
						  else if(relation.getChemical().text.toLowerCase().equals(previous.text.toLowerCase()))  {
							  b = 2;
							  break;
						  }
					}
					
					features.add("GLOFEA10#"+b);	
					
				} 
			}
			
					
			
		} else {
			
			{ // former -> B and latter
				Entity B = Util.getClosestCoorEntity(tokens, latter, predict);
				if(B != null) {
					if(!B.type.equals(former.type)) {
						Entity chemical = null;
						Entity disease = null;
						if(former.type.equals("Chemical")) {
							chemical = former;
							disease = B;
						} else {
							chemical = B;
							disease = former;
						}
						int b = 0;
						for(RelationEntity relation:predict.relations) {
							  if(relation.getDisease().text.toLowerCase().equals(disease.text.toLowerCase()) 
								&& relation.getChemical().text.toLowerCase().equals(chemical.text.toLowerCase())) {
								    b = 1;
							  		break;
							  }
						 }
						
						features.add("GLOFEA6#"+b);
					} 
				} 
			}
			
			
			{ // A and former -> latter
				Entity A = Util.getClosestCoorEntity(tokens, former, predict);
				if(A != null) {
					if(!A.type.equals(latter.type)) {
						Entity chemical = null;
						Entity disease = null;
						if(latter.type.equals("Chemical")) {
							chemical = latter;
							disease = A;
						} else {
							chemical = A;
							disease = latter;
						}
						int  b = 0;
						for(RelationEntity relation:predict.relations) {
							if(relation.getDisease().text.toLowerCase().equals(disease.text.toLowerCase()) 
									&& relation.getChemical().text.toLowerCase().equals(chemical.text.toLowerCase())) {
								  b=1;
								  break;
							  }
						  }
						
						
						features.add("GLOFEA7#"+b);
					} 
	
				} 
			}
			
			{   // former latter has been in ADE
				Entity chemical = null;
				Entity disease = null;
				if(latter.type.equals("Chemical")) {
					chemical = latter;
					disease = former;
				} else {
					chemical = former;
					disease = latter;
				}
				int b = 0;
				for(RelationEntity relation:predict.relations) {
					if(relation.getDisease().text.toLowerCase().equals(disease.text.toLowerCase()) 
							&& relation.getChemical().text.toLowerCase().equals(chemical.text.toLowerCase())) {
							b = 1;
							
					  }
				  }
				
				features.add("GLOFEA8#"+b);
					
			}
			
			
			
		}
		
		return features;
	}
	
	// Given the tokens of a sentence and the index of current token, generate a example filled with
	// all features but labels not 
	public Example getExampleFeatures(List<CoreLabel> tokens, int idx, boolean bRelation,
			Entity former, Entity latter, Tool tool, Prediction predict) throws Exception {
		Example example = new Example(bRelation);
		
		if(!bRelation) {
			// current word
			example.featureIdx.add(getWordID(tokens.get(idx)));
			
			// words before the current word, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getWordID(tokens.get(idxBefore)));
				} else {
					example.featureIdx.add(getPaddingID());
				}
			}
			// words after the current word, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getWordID(tokens.get(idxAfter)));
				} else {
					example.featureIdx.add(getPaddingID());
				}
			}
			
			// current pos
			example.featureIdx.add(getPosID(tokens.get(idx).tag()));
			
			// context pos
			for(int i=0;i<1;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getPosID(tokens.get(idxBefore).tag()));
				} else {
					example.featureIdx.add(getPosID(Parameters.PADDING));
				}
			}
			for(int i=0;i<1;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getPosID(tokens.get(idxAfter).tag()));
				} else {
					example.featureIdx.add(getPosID(Parameters.PADDING));
				}
			}
			
			// current prefix and suffix
			example.featureIdx.add(getPreSuffixID(getPrefix(tokens.get(idx))));
			example.featureIdx.add(getPreSuffixID(getSuffix(tokens.get(idx))));
			
			// context prefix and suffix
			for(int i=0;i<2;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getPreSuffixID(getPrefix(tokens.get(idxBefore))));
					example.featureIdx.add(getPreSuffixID(getSuffix(tokens.get(idxBefore))));
				} else {
					example.featureIdx.add(getPreSuffixID(Parameters.PADDING));
					example.featureIdx.add(getPreSuffixID(Parameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getPreSuffixID(getPrefix(tokens.get(idxAfter))));
					example.featureIdx.add(getPreSuffixID(getSuffix(tokens.get(idxAfter))));
				} else {
					example.featureIdx.add(getPreSuffixID(Parameters.PADDING));
					example.featureIdx.add(getPreSuffixID(Parameters.PADDING));
				}
			}
			
			// current brown
			example.featureIdx.add(getBrownID(getBrown(tokens.get(idx), tool)));
			
			// context brown
			for(int i=0;i<2;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getBrownID(getBrown(tokens.get(idxBefore), tool)));
				} else {
					example.featureIdx.add(getBrownID(Parameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getBrownID(getBrown(tokens.get(idxAfter), tool)));
				} else {
					example.featureIdx.add(getBrownID(Parameters.PADDING));
				}
			}
			
			// current synset
			example.featureIdx.add(getSynsetID(getSynset(tokens.get(idx), tool)));
			
			// context synset
			for(int i=0;i<2;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getSynsetID(getSynset(tokens.get(idxBefore), tool)));
				} else {
					example.featureIdx.add(getSynsetID(Parameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getSynsetID(getSynset(tokens.get(idxAfter), tool)));
				} else {
					example.featureIdx.add(getSynsetID(Parameters.PADDING));
				}
			}
			
			// current hyper
			example.featureIdx.add(getHyperID(getHyper(tokens.get(idx), tool)));
			
			// current dict
			example.featureIdx.add(getDictID(getDict(tokens.get(idx), tool)));
			// context dict
			for(int i=0;i<2;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getDictID(getDict(tokens.get(idxBefore), tool)));
				} else {
					example.featureIdx.add(getDictID(Parameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getDictID(getDict(tokens.get(idxAfter), tool)));
				} else {
					example.featureIdx.add(getDictID(Parameters.PADDING));
				}
			}
			
			// global features
			
			{ // is token in any ADE
				String current = tokens.get(idx).word().toLowerCase();
				boolean b = false; 
				int isIn = 0; // 0-not, 1-disease, 2-chemical
				  for(RelationEntity relation:predict.relations) {
					  for(int i=relation.getDisease().start;i<=relation.getDisease().end;i++) {
						  if(tokens.get(i).word().toLowerCase().equals(current)) {
							  isIn = 1;
							  b = true;
							  
							  break;
						  }
					  }
					  
					  if(isIn == 0) {
						  for(int i=relation.getChemical().start;i<=relation.getChemical().end;i++) {
							  if(tokens.get(i).word().toLowerCase().equals(current)) {
								  isIn = 2;
								  b = true;
								  
								  break;
							  }
						  }
					  }
					  
					  if(isIn != 0)
						  break;
						  
				  }
				  

				example.featureIdx.add(getGlobalFeatureID("GLOFEA2#"+isIn));	 
				
			}
			
			{ // is token's brown in any ADE
				String tokBrown = getBrown(tokens.get(idx), tool);
				if(tokBrown != null) {
					boolean b = false;
					int isIn = 0; // 0-not, 1-disease, 2-chemical
					for(RelationEntity relation:predict.relations) {
						
						  for(int i=relation.getDisease().start;i<=relation.getDisease().end;i++) {
							  String brownDisease = getBrown(tokens.get(i), tool);
							  if(brownDisease==null)
								  continue;
							  
							  if(tokBrown.equals(brownDisease)) {
								  isIn = 1;
								  b = true;
								  
								  break;
							  }
						  }
						  
						  if(isIn == 0) {
							  for(int i=relation.getChemical().start;i<=relation.getChemical().end;i++) {
								  String brownChem = getBrown(tokens.get(i), tool);
								  if(brownChem==null)
									  continue;
								  
								  if(tokBrown.equals(brownChem)) {
									  isIn = 2;
									  b = true;
									  
									  break;
								  }
							  }
						  }
						
						
						  if(isIn != 0)
							  break;
								
					}
					
					example.featureIdx.add(getGlobalFeatureID("GLOFEA3#"+isIn));	
				} else
					example.featureIdx.add(-1);	
			}
				

			
			
			{ // is token's Synset in any ADE
				String toksynset = getSynset(tokens.get(idx), tool);
				if(toksynset != null) {
					boolean b = false;
					int isIn = 0; // 0-not, 1-disease, 2-chemical
					for(RelationEntity relation:predict.relations) {
						
						  for(int i=relation.getDisease().start;i<=relation.getDisease().end;i++) {
							  String synsetDisease = getSynset(tokens.get(i), tool);
							  if(synsetDisease==null)
								  continue;
							  
							  if(toksynset.equals(synsetDisease)) {
								  isIn = 1;
								  b = true;
								  
								  break;
							  }
						  }
						  
						  if(isIn == 0) {
							  for(int i=relation.getChemical().start;i<=relation.getChemical().end;i++) {
								  String synsetChem = getSynset(tokens.get(i), tool);
								  if(synsetChem==null)
									  continue;
								  
								  if(toksynset.equals(synsetChem)) {
									  isIn = 2;
									  b = true;
									  
									  break;
								  }
							  }
						  }
						
						
						  if(isIn != 0)
							  break;
								
					}
					
					example.featureIdx.add(getGlobalFeatureID("GLOFEA4#"+isIn));	
				} else
					example.featureIdx.add(-1);
				

			}
			
			
			{ // A -> B and tok
				Entity B = Util.getClosestCoorEntity(tokens, idx, predict);
				if(B != null) {
					
					int b = 0;
					  for(RelationEntity relation:predict.relations) {
						  if(relation.getDisease().text.toLowerCase().equals(B.text.toLowerCase())) {
							  b = 1;
							  break;
						  }
						  else if(relation.getChemical().text.toLowerCase().equals(B.text.toLowerCase()))  {
							  b = 2;
							  break;
						  }
					  }
					  
					  example.featureIdx.add(getGlobalFeatureID("GLOFEA5#"+b));
				} else
					example.featureIdx.add(-1);
			}
			
			
			{ 
				Entity previous = Util.getPreviousEntity(tokens, idx, predict);
				if(previous != null) {
					// whether the entity before tok is in relations
					int b = 0;
					for(RelationEntity relation:predict.relations) {
						  if(relation.getDisease().text.toLowerCase().equals(previous.text.toLowerCase())) {
							  b = 1;
							  break;
						  }
						  else if(relation.getChemical().text.toLowerCase().equals(previous.text.toLowerCase()))  {
							  b = 2;
							  break;
						  }
					}
					
					example.featureIdx.add(getGlobalFeatureID("GLOFEA10#"+b));	
					
				} else
					example.featureIdx.add(-1);
			}
			
			/** 
			 * The ones below are relation features.
			 */
			// words after the first entity
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			// words before the second entity
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			
			// entity type
			//example.featureIdx.add(-1);
			//example.featureIdx.add(-1);
			
			// entity wordnet
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			
			// global
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);			
					
			
		} else {
			// current word
			example.featureIdx.add(-1);
				
			// words before the former, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxBefore = former.start-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getWordID(tokens.get(idxBefore)));
				} else {
					example.featureIdx.add(getPaddingID());
				}
			}
			// words after the latter, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxAfter = latter.end+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getWordID(tokens.get(idxAfter)));
				} else {
					example.featureIdx.add(getPaddingID());
				}
			}
			
			// current pos
			example.featureIdx.add(-1);
			
			// context pos
			for(int i=0;i<1;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<1;i++) {
				example.featureIdx.add(-1);
			}
			
			// current prefix and suffix
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			
			// context prefix and suffix
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
				example.featureIdx.add(-1);
			}
			
			// current brown 
			example.featureIdx.add(-1);
			
			// context brown
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			
			// current synset
			example.featureIdx.add(-1);
			
			// context synset
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			
			// current hyper
			example.featureIdx.add(-1);
						
			// current dict
			example.featureIdx.add(-1);
			// context dict
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			
			// global feature
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			
			/** 
			 * The ones below are relation features.
			 */
			// words after the first entity
			for(int i=0;i<2;i++) {
				int idxAfter = former.end+1+i;
				if(idxAfter<=latter.start-1) {
					example.featureIdx.add(getWordID(tokens.get(idxAfter)));
				} else {
					example.featureIdx.add(getPaddingID());
				}
				
			}
			// words before the second entity
			for(int i=0;i<2;i++) {
				int idxBefore = latter.start-1-i;
				if(idxBefore>=former.end+1) {
					example.featureIdx.add(getWordID(tokens.get(idxBefore)));
				} else {
					example.featureIdx.add(getPaddingID());
				}
			}
			
			// entity type
			//example.featureIdx.add(getEntityTypeID(former));
			//example.featureIdx.add(getEntityTypeID(latter));
			
			// entity wordnet
			example.featureIdx.add(getSynsetID(getSynset(former.text, tool)));
			example.featureIdx.add(getSynsetID(getSynset(latter.text, tool)));
			example.featureIdx.add(getHyperID(getHyper(former.text, tool)));
			example.featureIdx.add(getHyperID(getHyper(latter.text, tool)));
			
			// global
			{ // former -> B and latter
				Entity B = Util.getClosestCoorEntity(tokens, latter, predict);
				if(B != null) {
					if(!B.type.equals(former.type)) {
						Entity chemical = null;
						Entity disease = null;
						if(former.type.equals("Chemical")) {
							chemical = former;
							disease = B;
						} else {
							chemical = B;
							disease = former;
						}
						int b = 0;
						for(RelationEntity relation:predict.relations) {
							  if(relation.getDisease().text.toLowerCase().equals(disease.text.toLowerCase()) 
								&& relation.getChemical().text.toLowerCase().equals(chemical.text.toLowerCase())) {
								    b = 1;
							  		break;
							  }
						 }
						
						example.featureIdx.add(getGlobalFeatureID("GLOFEA6#"+b));
					} else
						example.featureIdx.add(-1);
				} else
					example.featureIdx.add(-1);
			}
			
			
			{ // A and former -> latter
				Entity A = Util.getClosestCoorEntity(tokens, former, predict);
				if(A != null) {
					if(!A.type.equals(latter.type)) {
						Entity chemical = null;
						Entity disease = null;
						if(latter.type.equals("Chemical")) {
							chemical = latter;
							disease = A;
						} else {
							chemical = A;
							disease = latter;
						}
						int  b = 0;
						for(RelationEntity relation:predict.relations) {
							if(relation.getDisease().text.toLowerCase().equals(disease.text.toLowerCase()) 
									&& relation.getChemical().text.toLowerCase().equals(chemical.text.toLowerCase())) {
								  b=1;
								  break;
							  }
						  }
						
						
						example.featureIdx.add(getGlobalFeatureID("GLOFEA7#"+b));
					} else
						example.featureIdx.add(-1); 
	
				} else
					example.featureIdx.add(-1);
			}
			
			{   // former latter has been in ADE
				Entity chemical = null;
				Entity disease = null;
				if(latter.type.equals("Chemical")) {
					chemical = latter;
					disease = former;
				} else {
					chemical = former;
					disease = latter;
				}
				int b = 0;
				for(RelationEntity relation:predict.relations) {
					if(relation.getDisease().text.toLowerCase().equals(disease.text.toLowerCase()) 
							&& relation.getChemical().text.toLowerCase().equals(chemical.text.toLowerCase())) {
							b = 1;
							
					  }
				  }
				
				example.featureIdx.add(getGlobalFeatureID("GLOFEA8#"+b));
					
			}
									
			/*
			 * The following features are composite.
			 */
			// entity
			for(int i=former.start;i<=former.end;i++) {
				CoreLabel token = tokens.get(i);
				int embIdx = getWordID(token);
				example.formerIdx.add(embIdx);
			}
			for(int i=latter.start;i<=latter.end;i++) {
				CoreLabel token = tokens.get(i);
				int embIdx = getWordID(token);
				example.latterIdx.add(embIdx);
			}
			// sentence
			if(parameters.sentenceConvolution)
				fillSentenceIdx(tokens, former, latter, predict.entities, example);
			
		}
		
		return example;
	}
	
	public void fillSentenceIdx(List<CoreLabel> tokens, Entity former, Entity latter, List<Entity> entities,
			Example example) {
		// fill other entities with PADDING and fill the corresponding positions
		for(int i=0;i<tokens.size();i++) {
			if(Util.isInsideAEntity(tokens.get(i).beginPosition(), tokens.get(i).endPosition(), former)) {
				example.sentenceIdx.add(getWordID(tokens.get(i)));
				example.positionIdxFormer.add(getPositionID(0));
				example.positionIdxLatter.add(getPositionID(i-latter.start));
			} else if(Util.isInsideAEntity(tokens.get(i).beginPosition(), tokens.get(i).endPosition(), latter)) {
				example.sentenceIdx.add(getWordID(tokens.get(i)));
				example.positionIdxFormer.add(getPositionID(i-former.end));
				example.positionIdxLatter.add(getPositionID(0));
			} else if(-1 != Util.isInsideAGoldEntityAndReturnIt(tokens.get(i).beginPosition(), tokens.get(i).endPosition(), entities)) {
				example.sentenceIdx.add(getPaddingID());
			} else {
				example.sentenceIdx.add(getWordID(tokens.get(i)));
				if(i<former.start)
					example.positionIdxFormer.add(getPositionID(i-former.start));
				else
					example.positionIdxFormer.add(getPositionID(i-former.end));
				
				if(i<latter.start)
					example.positionIdxLatter.add(getPositionID(i-latter.start));
				else
					example.positionIdxLatter.add(getPositionID(i-latter.end));
			}

		}
	}
	
	public CoreLabel getHeadWord(Entity entity, List<CoreLabel> tokens) {
		return tokens.get(entity.end);
	}
	
	public int getGlobalFeatureID(String s) {
		return globalFeatureIDs.containsKey(s) ? globalFeatureIDs.get(s) : 
			globalFeatureIDs.get(Parameters.UNKNOWN);
	}
	
	@Override
	public int getPaddingID() {
		return wordIDs.get(Parameters.PADDING);
	}
		
	public int getWordID(CoreLabel token) {
		String temp = wordPreprocess(token, parameters);
		return wordIDs.containsKey(temp) ? wordIDs.get(temp) : wordIDs.get(Parameters.UNKNOWN);
			
	 }
	
	public int getPosID(String s) {
	      return posIDs.containsKey(s) ? posIDs.get(s) : posIDs.get(Parameters.UNKNOWN);
	  }
	
	public int getPreSuffixID(String s) {
		return presuffixIDs.containsKey(s) ? presuffixIDs.get(s) : presuffixIDs.get(Parameters.UNKNOWN);
	}
	
	public String getPrefix(CoreLabel token) {
		int len = token.lemma().length()>parameters.prefixLength ? parameters.prefixLength:token.lemma().length();
		return token.lemma().substring(0, len);
	}
	
	public String getSuffix(CoreLabel token) {
		int len = token.lemma().length()>parameters.prefixLength ? parameters.prefixLength:token.lemma().length();
		return token.lemma().substring(token.lemma().length()-len, token.lemma().length());
	}
	
	public String getBrown(CoreLabel token, Tool tool) {
		return tool.brownCluster.getPrefix(token.lemma());
	}
	
	public int getBrownID(String s) {

		return brownIDs.containsKey(s) ? brownIDs.get(s) : brownIDs.get(Parameters.UNKNOWN);
	}
	
	public String getSynset(CoreLabel token, Tool tool) {
		ISynset synset = WordNetUtil.getMostSynset(tool.dict, token.lemma(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostSynset(tool.dict, token.lemma(), POS.ADJECTIVE);
			if(synset!= null)
				return String.valueOf(synset.getID().getOffset());
			else
				return null;
		}
	}
	
	public String getSynset(String s, Tool tool) {
		ISynset synset = WordNetUtil.getMostSynset(tool.dict, s.toLowerCase(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostSynset(tool.dict, s.toLowerCase(), POS.ADJECTIVE);
			if(synset!= null)
				return String.valueOf(synset.getID().getOffset());
			else
				return null;
		}
	}
	
	public int getSynsetID(String s) {
		if(s==null)
			return synsetIDs.get(Parameters.UNKNOWN);
		else
			return synsetIDs.containsKey(s) ? synsetIDs.get(s) : synsetIDs.get(Parameters.UNKNOWN);
	}
	
	public String getHyper(CoreLabel token, Tool tool) {
		ISynset synset = WordNetUtil.getMostHypernym(tool.dict, token.lemma(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostHypernym(tool.dict, token.lemma(), POS.ADJECTIVE);
			if(synset != null)
				return String.valueOf(synset.getID().getOffset());
			else
				return null;
		}
	}
	
	public String getHyper(String s, Tool tool) {
		ISynset synset = WordNetUtil.getMostHypernym(tool.dict, s.toLowerCase(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostHypernym(tool.dict, s.toLowerCase(), POS.ADJECTIVE);
			if(synset != null)
				return String.valueOf(synset.getID().getOffset());
			else
				return null;
		}
	}
	
	public int getHyperID(String s) {
		if(s==null)
			return hyperIDs.get(Parameters.UNKNOWN);
		else
			return hyperIDs.containsKey(s) ? hyperIDs.get(s) : hyperIDs.get(Parameters.UNKNOWN);
	}
	
	public String getDict(CoreLabel token, Tool tool) {
		if(tool.humando.contains(token.word()) || tool.humando.contains(token.lemma()))
			return "disease";
		else if(tool.ctdmedic.contains(token.word()) || tool.ctdmedic.contains(token.lemma()))
			return "disease";
		
		if(tool.chemElem.containsCaseSensitive(token.word()))
			return "drug";
		else if(tool.drugbank.contains(token.word()) || tool.drugbank.contains(token.lemma()))
			return "drug";
		else if(tool.jochem.contains(token.word()) || tool.jochem.contains(token.lemma()))
			return "drug";
		else if(tool.ctdchem.contains(token.word()) || tool.ctdchem.contains(token.lemma()))
			return "drug";
		
		
		return "other";	
	}
	
	public int getDictID(String s) {
		if(s==null)
			return dictIDs.get(Parameters.UNKNOWN);
		else
			return dictIDs.containsKey(s) ? dictIDs.get(s) : dictIDs.get(Parameters.UNKNOWN);
	}
	
	public int getEntityTypeID(Entity entity) {
		return entitytypeIDs.containsKey(entity.type) ? entitytypeIDs.get(entity.type) : entitytypeIDs.get(Parameters.UNKNOWN);
	}
	
	public String getRelationDict(Entity former, Entity latter, Tool tool)  {
		if(former.type.equals("Chemical") && latter.type.equals("Disease") 
				&& tool.sider.contains(former.text, latter.text)) {
			return "ade";
		} else if(former.type.equals("Disease") && latter.type.equals("Chemical")
				&& tool.sider.contains(latter.text, former.text)) {
			return "ade";
		} else
			return null;
	}
	
	
	
	
	// check the wrong sequence [0,4,5] 3 when curTran is 3
	public boolean checkWrongState(Prediction prediction) {
		int position045 = -1;
		int positionOf1or2 = -1;
		for(int j=prediction.labels.size()-2;j>=0;j--) {
			if(prediction.labels.get(j)==1 || prediction.labels.get(j)==2)
				positionOf1or2 = j;
			else if(prediction.labels.get(j)==0 || prediction.labels.get(j)==4 || prediction.labels.get(j)==5)
				position045 = j;
			 
			
			if(position045!=-1 && positionOf1or2!=-1)
				break;
		}
		
		if(position045 < positionOf1or2) 
			return true;
		else
			return false;
	}
	
	public void softmaxScores(double[] scores1) {
		int optLabel1 = -1; 
        for (int i = 0; i < scores1.length; ++i) {
            if (optLabel1 < 0 || scores1[i] > scores1[optLabel1])
              optLabel1 = i;  
        }
        double sum1 = 0.0;
        double maxScore = scores1[optLabel1];
        for (int i = 0; i < scores1.length; ++i) {
        	scores1[i] = Math.exp(scores1[i] - maxScore);
            sum1 += scores1[i];
        }
        for (int i = 0; i < scores1.length; ++i) {
        	scores1[i] = scores1[i]/sum1;
        }
	}

	// Given a raw sentence, output the prediction
	public ADESentence decode(List<CoreLabel> tokens, Tool tool) throws Exception {
		Prediction prediction = new Prediction();
		for(int idx=0;idx<tokens.size();idx++) {
			// prepare the input for NN
			Example ex = getExampleFeatures(tokens, idx, false, null, null, tool, prediction);
			int transition = nn.giveTheBestChoice(ex);
			prediction.addLabel(transition, -1);
				
			// generate entities based on the latest label
			int curTran = prediction.labels.get(prediction.labels.size()-1);
			if(curTran==1) { // new chemical
				CoreLabel current = tokens.get(idx);
				  Entity chem = new Entity(null, Parameters.CHEMICAL, current.beginPosition(), 
						  current.word(), null);
				  chem.start = idx;
				  chem.end = idx;
				  prediction.entities.add(chem);
			} else if(curTran==2) {// new disease
				CoreLabel current = tokens.get(idx);
				  Entity disease = new Entity(null, Parameters.DISEASE, current.beginPosition(), 
						  current.word(), null);
				  disease.start = idx;
				  disease.end = idx;
				  prediction.entities.add(disease);
			} else if(curTran==3 && checkWrongState(prediction)) { // append the current entity
				Entity old = prediction.entities.get(prediction.entities.size()-1);
				CoreLabel current = tokens.get(idx);
				int whitespaceToAdd = current.beginPosition()-(old.offset+old.text.length());
				for(int j=0;j<whitespaceToAdd;j++)
					old.text += " ";
				old.text += current.word();
				old.end = idx;
			}
			
			// begin to predict relations
			int lastTran = prediction.labels.size()>=2 ? prediction.labels.get(prediction.labels.size()-2) : -1;
			// judge whether to generate relations
			if((lastTran==1 && curTran==1) || (lastTran==1 && curTran==2) || (lastTran==1 && curTran==0)
					|| (lastTran==2 && curTran==0) || (lastTran==2 && curTran==1) || (lastTran==2 && curTran==2)
					|| (lastTran==3 && curTran==0) || (lastTran==3 && curTran==1) || (lastTran==3 && curTran==2)
			) { 
				if((lastTran==3 && checkWrongState(prediction)) || lastTran==1 || lastTran==2) {
					// if curTran 1 or 2, the last entities should not be considered
					int latterIdx = (curTran==1 || curTran==2) ? prediction.entities.size()-2:prediction.entities.size()-1;
					Entity latter = prediction.entities.get(latterIdx);
					for(int j=0;j<latterIdx;j++) {
						Entity former = prediction.entities.get(j);
						if(latter.type.equals(former.type))
							continue;
						Example relationExample = getExampleFeatures(tokens, -1, true, former, latter, tool, prediction);
						transition = nn.giveTheBestChoice(relationExample);
						prediction.addLabel(transition,-1);

			            // generate relations based on the latest label
			            curTran = prediction.labels.get(prediction.labels.size()-1);
			        	if(curTran == 5) { // connect
							RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
							prediction.relations.add(relationEntity);
			        	}

					}
				}
				
			}
				
			
			
		}
		
		// when at the end of sentence, judge relation ignoring lastTran
		int curTran = prediction.labels.get(prediction.labels.size()-1);
		if((curTran==3 && checkWrongState(prediction)) || curTran==1 || curTran==2) {
			int latterIdx = prediction.entities.size()-1;
			Entity latter = prediction.entities.get(latterIdx);

			for(int j=0;j<latterIdx;j++) {
				Entity former = prediction.entities.get(j);
				if(latter.type.equals(former.type))
					continue;
				Example relationExample = getExampleFeatures(tokens, -1, true, former, latter, tool, prediction);
				int transition = nn.giveTheBestChoice(relationExample);
				prediction.addLabel(transition, -1);

	            // generate relations based on the latest label
	            curTran = prediction.labels.get(prediction.labels.size()-1);
	        	if(curTran == 5) { // connect
					RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
					prediction.relations.add(relationEntity);
	        	}

			}
			
		}
		
		
		// Prediction to ADESentence
		ADESentence predicted = new ADESentence();
		predicted.entities.addAll(prediction.entities);
		predicted.relaitons.addAll(prediction.relations);
		
		return predicted;
	}
	
	
	public List<CoreLabel> prepareNLPInfo(Tool tool, ADESentence sentence) {
		ArrayList<CoreLabel> tokens = tool.tokenizer.tokenize(sentence.offset, sentence.text);
		tool.tagger.tagCoreLabels(tokens);
		for(int i=0;i<tokens.size();i++)
			tool.morphology.stem(tokens.get(i));
		
		return tokens;
	}
	
	/*
	 * Given a word, we will transform it based on the "wordPreprocess". 
	 * Make sure call this function before send a word to NNADE.
	 */
	public static String wordPreprocess(CoreLabel token, Parameters parameters) {
		if(parameters.wordPreprocess == 0) {
			return normalize_to_lowerwithdigit(token.word());
		} else if(parameters.wordPreprocess == 1)
			return token.word(); 
		else if(parameters.wordPreprocess==2) {
			return token.lemma().toLowerCase();
		} else if(parameters.wordPreprocess==3) {
			return NNADE.pipe(token.lemma().toLowerCase());
		} else {
			return token.word().toLowerCase();
		}

	}
	
	public static String normalize_to_lowerwithdigit(String s)
	{
		String lowcase = "";
	  char [] chars = s.toCharArray();
	  for (int i = 0; i < chars.length; i++) {
	    if (Character.isDigit(chars[i])) {
	      lowcase = lowcase + "0";
	    } else if (Character.isLetter(chars[i])) {
	      if (Character.isLowerCase(chars[i]))
	      {
	        lowcase = lowcase + chars[i];
	      }
	      else
	      {
	        lowcase = lowcase + Character.toLowerCase(chars[i]) ;
	      }
	    }
	    else
	    {
	      lowcase = lowcase + chars[i];
	    }
	  }
	  return lowcase;
	}
	
	public static String pipe(String word) {
		char[] chs = word.toCharArray();
		for(int i=0;i<chs.length;i++) {
			if(CharCode.isNumber(chs[i]))
				chs[i] = '0';
			else if(Punctuation.isEnglishPunc(chs[i]))
				chs[i] = '*';
		}
		return new String(chs);
	}

	@Override
	public double[][] getE() {
		return E;
	}

	@Override
	public double[][] getEg2E() {
		// TODO Auto-generated method stub
		return eg2E;
	}
	


	@Override
	public int getPositionID(int position) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public List<String> getKnownWords() {
		// TODO Auto-generated method stub
		return knownWords;
	}

	

}


