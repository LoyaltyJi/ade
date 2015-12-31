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
import utils.ADESentence;
import utils.Abstract;

public class NNADE_word extends Father implements Serializable {

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
	
	// key-word, value-word ID (the row id of the embedding matrix)
	public TObjectIntHashMap<String> wordIDs;
	public TObjectIntHashMap<String> posIDs;
	public TObjectIntHashMap<String> presuffixIDs;
	public TObjectIntHashMap<String> brownIDs;
	public TObjectIntHashMap<String> synsetIDs;
	public TObjectIntHashMap<String> hyperIDs;
	public TObjectIntHashMap<String> dictIDs;
	public TObjectIntHashMap<String> entitytypeIDs;
	
	// only used when loading external embeddings
	public TObjectIntHashMap<String> embedID;
	public double[][] embeddings;

	// the embedding matrix, embedding numbers x embeddingSize
	public double[][] E; // word embedding can use pre-trained
	public double[][] eg2E;
	

	
	// Store the high-frequency token-position
	public TIntArrayList preComputed;
	
	public boolean debug;
	

	public NNADE_word(Parameters parameters) {
		
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
			
			NNADE_word nnade = new NNADE_word(parameters);
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

	    
	    // Generate word id which can be used in the embedding matrix
	    wordIDs = new TObjectIntHashMap<String>();
	    posIDs = new TObjectIntHashMap<>();
	    presuffixIDs = new TObjectIntHashMap<>();
	    brownIDs = new TObjectIntHashMap<>();
	    synsetIDs = new TObjectIntHashMap<>();
	    hyperIDs = new TObjectIntHashMap<>();
	    dictIDs = new TObjectIntHashMap<>();
	    entitytypeIDs = new TObjectIntHashMap<>();
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

	    System.out.println("#Word: " + knownWords.size());
	    System.out.println("#POS: " + knownPos.size());
	    System.out.println("#PreSuffix: " + knownPreSuffix.size());
	    System.out.println("#Brown: " + knownBrown.size());
	    System.out.println("#Synset: " + knownSynSet.size());
	    System.out.println("#Hyper: " + knownHyper.size());
	    System.out.println("#Dict: " + knownDict.size());
	    System.out.println("#Entity type: "+knownEntityType.size());
	    
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

		// generate training examples
		Counter<Integer> tokPosCount = new IntCounter<>();
		List<Example> exampleEntity = generateEntityTrainExamples(trainAbs, tokPosCount, tool);
		List<Example> exampleRelation = generateRelationTrainExamples(trainAbs, tokPosCount, tool);
		System.out.println("non-composite feature number: "+exampleEntity.get(0).featureIdx.size());
		// initialize preComputed
		preComputed = new TIntArrayList();
	    List<Integer> sortedTokens = Counters.toSortedList(tokPosCount, false);
	    List<Integer> sublist = sortedTokens.subList(0, Math.min(parameters.numPreComputed, sortedTokens.size()));
	    for(int tokPos : sublist) {
	    	preComputed.add(tokPos);
	    }
		
		// new a NN and initialize its weight
	    nn  = new NNSimple(parameters, this, preComputed, exampleEntity.get(0));
		nn.debug = debug;
		
		// train iteration
		long startTime = System.currentTimeMillis();
		BestPerformance best = new BestPerformance();
		
		for (int iter = 0; iter < parameters.maxIter; ++iter) {
			if(debug)
				System.out.println("##### Iteration " + iter);
			
			// mini-batch
			int batchSizeEntity = (int)(exampleEntity.size()*parameters.batchEntityPercent);
			if(batchSizeEntity == 0)
				batchSizeEntity++;
			List<Example> batchExampleEntity = Util.getRandomSubList(exampleEntity, batchSizeEntity);
			
			int batchSizeRelation = (int)(exampleRelation.size()*parameters.batchRelationPercent);
			if(batchSizeRelation == 0)
				batchSizeRelation++;
			List<Example> batchExampleRelation = Util.getRandomSubList(exampleRelation, batchSizeRelation);
			
			
			List<Example> examples = new ArrayList<>();
			examples.addAll(batchExampleEntity);
			examples.addAll(batchExampleRelation);
			if(debug)
				System.out.println("batch size: "+examples.size());
			
			//GradientKeeper keeper = nn.process(examples, null);
			GradientKeeper1 keeper = nn.process(examples, null);
			
			//nn.checkGradients(keeper, examples);
			nn.updateWeights(keeper);
			
			if(debug)
				System.out.println("Elapsed Time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " (s)");
			
			if (iter>0 && iter % parameters.evalPerIter == 0) {
				evaluate(tool, devAbs, testAbs, modelFile, best);
			}			
		}
		
		evaluate(tool, devAbs, testAbs, modelFile, best);
		
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
			List<Abstract> testAbs, String modelFile, BestPerformance best)
			throws Exception {
		// Redo precomputation with updated weights. This is only
        // necessary because we're updating weights -- for normal
        // prediction, we just do this once 
        nn.preCompute();

        DecodeStatistic stat = new DecodeStatistic();
        for(Abstract devAb:devAbs) {
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

        	}
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
          ObjectSerializer.writeObjectToFile(this, modelFile);
          
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
	
	public List<Example> generateEntityTrainExamples(List<Abstract> trainAbs, Counter<Integer> tokPosCount, Tool tool)
			throws Exception {
		List<Example> ret = new ArrayList<>();
				
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				// resort the entities in the sentence
				List<Entity> entities = Util.resortEntity(sentence);
				
				// for each token, we generate an entity example
				for(int idx=0;idx<tokens.size();idx++) {
					Example example = getExampleFeatures(tokens, idx, false, null, null, tool, entities);
					double[] goldLabel = {0,0,0,0,0,0};
					CoreLabel token = tokens.get(idx);
					
					int index = Util.isInsideAGoldEntityAndReturnIt(token.beginPosition(), token.endPosition(), entities);
					
					if(index == -1) {
						// other
						goldLabel[0] = 1;
					} else {
						Entity gold = entities.get(index);
						if(Util.isFirstWordOfEntity(gold, token)) {
							if(gold.type.equals(Parameters.CHEMICAL)) {
								// new chemical
								goldLabel[1] = 1;
							} else {
								// new disease
								goldLabel[2] = 1;
							}
						} else {
							// append
							goldLabel[3] = 1;
						}
						
						
					}
					
					example.label = goldLabel;
					ret.add(example);
					
					for(int j=0; j<example.featureIdx.size(); j++)
						if(example.featureIdx.get(j) != -1)
							tokPosCount.incrementCount(example.featureIdx.get(j)*example.featureIdx.size()+j);
					
				}
				
				
			}
		}
		
		return ret;
	}
	
	public List<Example> generateRelationTrainExamples(List<Abstract> trainAbs, Counter<Integer> tokPosCount, Tool tool)
			throws Exception {
		List<Example> ret = new ArrayList<>();
				
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				// resort the entities in the sentence
				List<Entity> entities = Util.resortEntity(sentence);
				// fill 'start' and 'end' of the entities
				Util.fillEntity(entities, tokens);
				
				// for each entity pair, we generate a relation example
				for(int i=0;i<entities.size();i++) {
					Entity latter = entities.get(i);
					for(int j=0;j<i;j++) {
						Entity former = entities.get(j);
						
						Example example = getExampleFeatures(tokens, -1, true, former, latter, tool, entities);
						double[] goldLabel = {0,0,0,0,0,0};
						
						RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
						if(sentence.relaitons.contains(tempRelation)) {
							// connect
							goldLabel[5] = 1;
						} else {
							// not connect
							goldLabel[4] = 1;
						}
						example.label = goldLabel;
						ret.add(example);
						
						for(int k=0; k<example.featureIdx.size(); k++)
							if(example.featureIdx.get(k) != -1)
								tokPosCount.incrementCount(example.featureIdx.get(k)*example.featureIdx.size()+k);
					}
				}
				
	
			}
		}
		
		return ret;
	}
	
	// Given the tokens of a sentence and the index of current token, generate a example filled with
	// all features but labels not 
	public Example getExampleFeatures(List<CoreLabel> tokens, int idx, boolean bRelation,
			Entity former, Entity latter, Tool tool, List<Entity> entities) throws Exception {
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
				fillSentenceIdx(tokens, former, latter, entities, example);
			
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
			Example ex = getExampleFeatures(tokens, idx, false, null, null, tool, null);
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
						Example relationExample = getExampleFeatures(tokens, idx, true, former, latter, tool, prediction.entities);
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
				Example relationExample = getExampleFeatures(tokens, tokens.size()-1, true, former, latter, tool, prediction.entities);
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
	
	public ADESentence decode_old(List<CoreLabel> tokens, DecodeStatistic stat, Tool tool) throws Exception {
		ADESentence predicted = new ADESentence();
		
		List<Entity> tempEntities = new ArrayList<>();
		/*
		 * If the transition sequence is 1 0 3 3, we will have a bug when append.
		 * Add a flag 'newed' to indicate an entity begin(true) or end(false).
		 */
		boolean newed = false;
		int lastTran = -1;
		int curTran = -1;
		for(int idx=0;idx<tokens.size();idx++) {
			// prepare the input for NN
			Example ex = getExampleFeatures(tokens, idx, false, null, null, tool, null);
			// get the transition given by NN
			// other, newChemical, newDisease, append, notConnect, connect  
			lastTran = curTran;
			curTran = nn.giveTheBestChoice(ex);
			stat.total++;
			
			// predict the entity based on the transition
			if((lastTran==0 && curTran==0) || (lastTran==1 && curTran==0) || (lastTran==2 && curTran==0)
				|| (lastTran==3 && curTran==0) || (lastTran==4 && curTran==0) || (lastTran==5 && curTran==0))
			{ // no entity, just add a one-length non-entity segment
				  
			} else if((lastTran==0 && curTran==1) || (lastTran==1 && curTran==1) || (lastTran==2 && curTran==1)
				|| (lastTran==3 && curTran==1) || (lastTran==4 && curTran==1) || (lastTran==5 && curTran==1)
				|| (lastTran==-1 && curTran==1))
			{ // new chemical
				CoreLabel current = tokens.get(idx);
				  Entity chem = new Entity(null, Parameters.CHEMICAL, current.beginPosition(), 
						  current.word(), null);
				  chem.start = idx;
				  chem.end = idx;
				  tempEntities.add(chem);
				  newed = true;
			} else if((lastTran==0 && curTran==2) || (lastTran==1 && curTran==2) || (lastTran==2 && curTran==2)
				|| (lastTran==3 && curTran==2) || (lastTran==4 && curTran==2) || (lastTran==5 && curTran==2) 
				|| (lastTran==-1 && curTran==2))
			{// new disease
				CoreLabel current = tokens.get(idx);
				  Entity disease = new Entity(null, Parameters.DISEASE, current.beginPosition(), 
						  current.word(), null);
				  disease.start = idx;
				  disease.end = idx;
				  tempEntities.add(disease);
				  newed = true;
			} else if((lastTran==1 && curTran==3) || (lastTran==2 && curTran==3) || (lastTran==3 && curTran==3))
			{ // append the current entity
				if(newed == true) {
					Entity old = tempEntities.get(tempEntities.size()-1);
					CoreLabel current = tokens.get(idx);
					int whitespaceToAdd = current.beginPosition()-(old.offset+old.text.length());
					for(int j=0;j<whitespaceToAdd;j++)
						old.text += " ";
					old.text += current.word();
					old.end = idx;
				}
				
			} else if((lastTran==0 && curTran==3) || (lastTran==0 && curTran==4) || (lastTran==0 && curTran==5)
				|| (lastTran==1 && curTran==4) || (lastTran==1 && curTran==5) || (lastTran==2 && curTran==4) ||
				(lastTran==2 && curTran==5) || (lastTran==3 && curTran==4) || (lastTran==3 && curTran==5) ||
				(lastTran==4 && curTran==4) || (lastTran==4 && curTran==5) || (lastTran==5 && curTran==4) ||
				(lastTran==5 && curTran==5) || (lastTran==4 && curTran==3) || (lastTran==5 && curTran==3)) {
				/*
				 * wrong status
				 * if last=other, current=append, 
				 * or if current=4 or 5
				 * or if last = 4 or 5, current = append
				 */
				stat.wrong++;
			} else { // other but not wrong status
				
			}
			
			// if an entity ends, we should predict the relation between it and all the entities before it.
			if((lastTran==1 && curTran==1) || (lastTran==1 && curTran==2) || (lastTran==1 && curTran==0)
				|| (lastTran==2 && curTran==0) || (lastTran==2 && curTran==1) || (lastTran==2 && curTran==2)
				|| (lastTran==3 && curTran==0) || (lastTran==3 && curTran==1) || (lastTran==3 && curTran==2))
			{
				// If the transition sequence is 0 3 0, we will have a bug here without the if statement.
				if(newed == true) {
					Entity latter = tempEntities.get(tempEntities.size()-1);
					for(int j=0;j<tempEntities.size()-1;j++) {
						Entity former = tempEntities.get(j);
						Example relationExample = getExampleFeatures(tokens, idx, true, former, latter, tool, tempEntities);
						lastTran = curTran;
						curTran = nn.giveTheBestChoice(relationExample);
						stat.total++;
						
						if(curTran == 4) { // not connect
							
						} else if(curTran == 5) { // connect
							RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
							predicted.relaitons.add(relationEntity);
						} else {
							stat.wrong++;
						}
						
					}
				}
				newed = false;
			}
			
			
		}
		
		
		predicted.entities.addAll(tempEntities);
				
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
			return NNADE_word.pipe(token.lemma().toLowerCase());
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



