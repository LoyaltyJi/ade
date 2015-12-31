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


public class Combine implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6814631308712316743L;
	public CombineNN nn;
	public CombineParameters parameters;
	// dictionary
	public List<String> knownWords1;
	public List<String> knownWords2;
	public List<String> knownPos;
	public List<String> knownPreSuffix;
	public List<String> knownBrown;
	public List<String> knownSynSet;
	public List<String> knownHyper;
	public List<String> knownDict;
	public List<String> knownEntityType;
	
	// key-word, value-word ID (the row id of the embedding matrix)
	public TObjectIntHashMap<String> wordIDs1;
	public TObjectIntHashMap<String> wordIDs2;
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
	

	public Combine(CombineParameters parameters) {
		
		this.parameters = parameters;
	}

	public static void main(String[] args) throws Exception {
		FileInputStream fis = new FileInputStream(args[0]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
				
		CombineParameters parameters = new CombineParameters(properties);
		parameters.printParameters();
		
				
		File fAbstractDir = new File(PropertiesUtils.getString(properties, "corpusDir", ""));
		File groupFile = new File(PropertiesUtils.getString(properties, "groupFile", ""));
		String modelFile = PropertiesUtils.getString(properties, "modelFile", "");
		String embedFile = PropertiesUtils.getString(properties, "embedFile", "");
		
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
		
				
		Word2Vec w2v = new Word2Vec();
		if(embedFile != null && !embedFile.isEmpty()) {
			long startTime = System.currentTimeMillis();
			w2v.loadModel(embedFile, true);
			System.out.println("Load main pretrained embeddings using " + ((System.currentTimeMillis()-startTime) / 1000.0)+"s");
		} else {
			throw new Exception();
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
		
		
		List<BestPerformance> bestAll = new ArrayList<>();
		for(int i=0;i<groups.size();i++) {
			Set<String> group = groups.get(i);
			List<Abstract> trainAb = new ArrayList<>();
			List<Abstract> testAb = new ArrayList<>();
			for(File abstractFile:fAbstractDir.listFiles()) {
				Abstract ab = (Abstract)ObjectSerializer.readObjectFromFile(abstractFile.getAbsolutePath());
				if(group.contains(ab.id)) {
					// test abstract
					testAb.add(ab);
				} else {
					// train abstract
					trainAb.add(ab);
				}
				

			}
			
			Combine nnade = new Combine(parameters);
			nnade.debug = Boolean.parseBoolean(args[1]);
			//nnade.brownCluster = brown;
			//nnade.wordnet = dict;
			// save model for each group
			System.out.println(CombineParameters.SEPARATOR+" group "+i);
			BestPerformance best = nnade.trainAndTest(trainAb, testAb,modelFile+i, embedFile, 
					tool, w2v);
			bestAll.add(best);
		}
		
		// macro performance
		double macroP_Entity = 0;
		double macroR_Entity = 0;
		double macroF1_Entity = 0;
		double macroP_Relation = 0;
		double macroR_Relation = 0;
		double macroF1_Relation = 0;
		double fold = bestAll.size();
		for(BestPerformance best:bestAll) {
			macroP_Entity += best.pEntity/fold; 
			macroR_Entity += best.rEntity/fold;
			macroP_Relation += best.pRelation/fold;
			macroR_Relation += best.rRelation/fold;
		}
		macroF1_Entity = Evaluater.getFMeasure(macroP_Entity, macroR_Entity, 1);
		macroF1_Relation = Evaluater.getFMeasure(macroP_Relation, macroR_Relation, 1);
		
		System.out.println("macro entity precision\t"+macroP_Entity);
        System.out.println("macro entity recall\t"+macroR_Entity);
        System.out.println("macro entity f1\t"+macroF1_Entity);
        System.out.println("macro relation precision\t"+macroP_Relation);
        System.out.println("macro relation recall\t"+macroR_Relation);
        System.out.println("macro relation f1\t"+macroF1_Relation);
	}
	
	public BestPerformance trainAndTest(List<Abstract> trainAbs, List<Abstract> testAbs, String modelFile, 
			String embedFile, Tool tool, Word2Vec w2v) 
		throws Exception {
		// generate alphabet
		List<String> word1 = new ArrayList<>();
		List<String> word2 = new ArrayList<>();
		List<String> pos = new ArrayList<>();	
		List<String> presuffix = new ArrayList<>();
		List<String> brown = new ArrayList<>();
		List<String> synset = new ArrayList<>();
		List<String> hyper = new ArrayList<>();
		List<String> dict = new ArrayList<>();
		List<String> entityType = new ArrayList<>();
		for(int i=1;i<=parameters.wordCutOff1+1;i++) {
			entityType.add("Disease");
			entityType.add("Chemical");
		}
		
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				
				for(CoreLabel token:tokens) {
					
					word1.add(wordPreprocess(token, parameters));
					word2.add(wordPreprocess(token, parameters));
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
		

		// add the alphabet of the test set
		for(Abstract ab:testAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				
				for(CoreLabel token:tokens) {
					word2.add(wordPreprocess(token, parameters));
				}
				
			}
		}
			
			
		knownWords1 = Util.generateDict(word1, parameters.wordCutOff1);
	    knownWords1.add(0, CombineParameters.UNKNOWN);
	    knownWords1.add(1, CombineParameters.PADDING);
	    knownWords2 = Util.generateDict(word2, parameters.wordCutOff2);
	    knownWords2.add(0, CombineParameters.UNKNOWN);
	    knownWords2.add(1, CombineParameters.PADDING);
	    
	    knownPos = Util.generateDict(pos, parameters.wordCutOff1);
	    knownPos.add(0, CombineParameters.UNKNOWN);
	    knownPos.add(1, CombineParameters.PADDING);
	    knownPreSuffix = Util.generateDict(presuffix, parameters.wordCutOff1);
	    knownPreSuffix.add(0, CombineParameters.UNKNOWN);
	    knownPreSuffix.add(1, CombineParameters.PADDING);
	    knownBrown = Util.generateDict(brown, parameters.wordCutOff1);
	    knownBrown.add(0, CombineParameters.UNKNOWN);
	    knownBrown.add(1, CombineParameters.PADDING);
	    knownSynSet = Util.generateDict(synset, parameters.wordCutOff1);
	    knownSynSet.add(0, CombineParameters.UNKNOWN);
	    knownSynSet.add(1, CombineParameters.PADDING);
	    knownHyper = Util.generateDict(hyper, parameters.wordCutOff1);
	    knownHyper.add(0, CombineParameters.UNKNOWN);
	    knownHyper.add(1, CombineParameters.PADDING);
	    knownDict = Util.generateDict(dict, parameters.wordCutOff1);
	    knownDict.add(CombineParameters.UNKNOWN);
	    knownDict.add(CombineParameters.PADDING);
	    knownEntityType = Util.generateDict(entityType, parameters.wordCutOff1);
	    knownEntityType.add(CombineParameters.UNKNOWN);

	    
	    // Generate word id which can be used in the embedding matrix
	    wordIDs1 = new TObjectIntHashMap<>();
	    wordIDs2 = new TObjectIntHashMap<>();
	    
	    posIDs = new TObjectIntHashMap<>();
	    presuffixIDs = new TObjectIntHashMap<>();
	    brownIDs = new TObjectIntHashMap<>();
	    synsetIDs = new TObjectIntHashMap<>();
	    hyperIDs = new TObjectIntHashMap<>();
	    dictIDs = new TObjectIntHashMap<>();
	    entitytypeIDs = new TObjectIntHashMap<>();
	    int m = 0;
	    for (String temp : knownWords1)
	      wordIDs1.put(temp, (m++));
	    for (String temp : knownWords2)
		  wordIDs2.put(temp, (m++));
	    
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

	    
	    E = new double[m][parameters.embeddingSize];
		eg2E = new double[E.length][E[0].length];
		
		// for 1, randowm
		randomInitialEmbedding(knownWords1, wordIDs1, E);
		// for 2, emb
		TIntArrayList uninitialIds = new TIntArrayList();
		int unknownID = -1;
		double sum[] = new double[parameters.embeddingSize];
		int count = 0;
		for (int i = 0; i < knownWords2.size(); ++i) {
			if(knownWords2.get(i).equals(CombineParameters.UNKNOWN)) {
				unknownID = wordIDs2.get(knownWords2.get(i));
				continue;
			}
			
		      String str = knownWords2.get(i);
		      int id = wordIDs2.get(str);
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
		if(count==0)
			count=1;
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
	    nn  = new CombineNN(parameters, this, preComputed, exampleEntity.get(0));
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
				evaluate(tool, testAbs, modelFile, best);
			}			
		}
		
		evaluate(tool, testAbs, modelFile, best);
		
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
			if(known.get(i).equals(CombineParameters.UNKNOWN)) {
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
		if(count==0)
			count=1;
		for (int idx = 0; idx < parameters.embeddingSize; idx++) {
		   emb[unknownID][idx] = sum[idx] / count;
		}
	}
	
	// Evaluate with the test set, and if the f1 is higher than bestF1, save the model
	public void evaluate(Tool tool, List<Abstract> testAbs, String modelFile, BestPerformance best)
			throws Exception {
		// Redo precomputation with updated weights. This is only
        // necessary because we're updating weights -- for normal
        // prediction, we just do this once 
        nn.preCompute();

        DecodeStatistic stat = new DecodeStatistic();
        for(Abstract testAb:testAbs) {
        	for(ADESentence gold:testAb.sentences) {
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
        
        System.out.println(CombineParameters.SEPARATOR);
        System.out.println("transiton wrong rate "+stat.getWrongRate());
        double pEntity = stat.getEntityPrecision();
        System.out.println("entity precision\t"+pEntity);
        double rEntity = stat.getEntityRecall();
        System.out.println("entity recall\t"+rEntity);
        double f1Entity = stat.getEntityF1();
        System.out.println("entity f1\t"+f1Entity);
        double pRelation = stat.getRelationPrecision();
        System.out.println("relation precision\t"+pRelation);
        double rRelation = stat.getRelationRecall();
        System.out.println("relation recall\t"+rRelation);
        double f1Relation = stat.getRelationF1();
        System.out.println("relation f1\t"+f1Relation);
        System.out.println(CombineParameters.SEPARATOR);

        	
        if ((f1Relation > best.f1Relation) || (f1Relation==best.f1Relation && f1Entity>best.f1Entity)) {
        //if ((f1Entity > best.f1Entity)) {
          System.out.printf("Current Exceeds the best! Saving model file %s\n", modelFile);
          best.pEntity = pEntity;
          best.rEntity = rEntity;
          best.f1Entity = f1Entity;
          best.pRelation = pRelation;
          best.rRelation = rRelation;
          best.f1Relation = f1Relation;
          //ObjectSerializer.writeObjectToFile(this, modelFile);
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
							if(gold.type.equals(CombineParameters.CHEMICAL)) {
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
						
						RelationEntity tempRelation = new RelationEntity(CombineParameters.RELATION, former, latter);
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
			example.featureIdx.add(getWordID1(tokens.get(idx)));
			example.featureIdx.add(getWordID2(tokens.get(idx)));
			
			// words before the current word, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getWordID1(tokens.get(idxBefore)));
					example.featureIdx.add(getWordID2(tokens.get(idxBefore)));
				} else {
					example.featureIdx.add(getPaddingID1());
					example.featureIdx.add(getPaddingID2());
				}
			}
			// words after the current word, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getWordID1(tokens.get(idxAfter)));
					example.featureIdx.add(getWordID2(tokens.get(idxAfter)));
				} else {
					example.featureIdx.add(getPaddingID1());
					example.featureIdx.add(getPaddingID2());
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
					example.featureIdx.add(getPosID(CombineParameters.PADDING));
				}
			}
			for(int i=0;i<1;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getPosID(tokens.get(idxAfter).tag()));
				} else {
					example.featureIdx.add(getPosID(CombineParameters.PADDING));
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
					example.featureIdx.add(getPreSuffixID(CombineParameters.PADDING));
					example.featureIdx.add(getPreSuffixID(CombineParameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getPreSuffixID(getPrefix(tokens.get(idxAfter))));
					example.featureIdx.add(getPreSuffixID(getSuffix(tokens.get(idxAfter))));
				} else {
					example.featureIdx.add(getPreSuffixID(CombineParameters.PADDING));
					example.featureIdx.add(getPreSuffixID(CombineParameters.PADDING));
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
					example.featureIdx.add(getBrownID(CombineParameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getBrownID(getBrown(tokens.get(idxAfter), tool)));
				} else {
					example.featureIdx.add(getBrownID(CombineParameters.PADDING));
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
					example.featureIdx.add(getSynsetID(CombineParameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getSynsetID(getSynset(tokens.get(idxAfter), tool)));
				} else {
					example.featureIdx.add(getSynsetID(CombineParameters.PADDING));
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
					example.featureIdx.add(getDictID(CombineParameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getDictID(getDict(tokens.get(idxAfter), tool)));
				} else {
					example.featureIdx.add(getDictID(CombineParameters.PADDING));
				}
			}
			
			/** 
			 * The ones below are relation features.
			 */
			// words after the first entity
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
				example.featureIdx.add(-1);
			}
			// words before the second entity
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
				example.featureIdx.add(-1);
			}
			
			// entity type
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			
			// entity wordnet
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			
					
			
		} else {
			// current word
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
				
			// words before the former, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxBefore = former.start-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getWordID1(tokens.get(idxBefore)));
					example.featureIdx.add(getWordID2(tokens.get(idxBefore)));
				} else {
					example.featureIdx.add(getPaddingID1());
					example.featureIdx.add(getPaddingID2());
				}
			}
			// words after the latter, but in the window
			for(int i=0;i<parameters.windowSize;i++) {
				int idxAfter = latter.end+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getWordID1(tokens.get(idxAfter)));
					example.featureIdx.add(getWordID2(tokens.get(idxAfter)));
				} else {
					example.featureIdx.add(getPaddingID1());
					example.featureIdx.add(getPaddingID2());
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
			
			/** 
			 * The ones below are relation features.
			 */
			// words after the first entity
			for(int i=0;i<2;i++) {
				int idxAfter = former.end+1+i;
				if(idxAfter<=latter.start-1) {
					example.featureIdx.add(getWordID1(tokens.get(idxAfter)));
					example.featureIdx.add(getWordID2(tokens.get(idxAfter)));
				} else {
					example.featureIdx.add(getPaddingID1());
					example.featureIdx.add(getPaddingID2());
				}
				
			}
			// words before the second entity
			for(int i=0;i<2;i++) {
				int idxBefore = latter.start-1-i;
				if(idxBefore>=former.end+1) {
					example.featureIdx.add(getWordID1(tokens.get(idxBefore)));
					example.featureIdx.add(getWordID2(tokens.get(idxBefore)));
				} else {
					example.featureIdx.add(getPaddingID1());
					example.featureIdx.add(getPaddingID2());
				}
			}
			
			// entity type
			example.featureIdx.add(getEntityTypeID(former));
			example.featureIdx.add(getEntityTypeID(latter));
			
			// entity wordnet
			example.featureIdx.add(getSynsetID(getSynset(former.text, tool)));
			example.featureIdx.add(getSynsetID(getSynset(latter.text, tool)));
			example.featureIdx.add(getHyperID(getHyper(former.text, tool)));
			example.featureIdx.add(getHyperID(getHyper(latter.text, tool)));
			
									
			/*
			 * The following features are composite.
			 */
			// entity
			for(int i=former.start;i<=former.end;i++) {
				CoreLabel token = tokens.get(i);
				example.formerIdx.add(getWordID1(token));
				example.formerIdx.add(getWordID2(token));
			}
			for(int i=latter.start;i<=latter.end;i++) {
				CoreLabel token = tokens.get(i);
				example.latterIdx.add(getWordID1(token));
				example.latterIdx.add(getWordID2(token));
			}
			
			
		}
		
		return example;
	}
	
	
	
	public CoreLabel getHeadWord(Entity entity, List<CoreLabel> tokens) {
		return tokens.get(entity.end);
	}
	
	public int getPaddingID1() {
		return wordIDs1.get(CombineParameters.PADDING);
	}
	
	public int getPaddingID2() {
		return wordIDs2.get(CombineParameters.PADDING);
	}
		
	public int getWordID1(CoreLabel token) {
		String temp = wordPreprocess(token, parameters);
		return wordIDs1.containsKey(temp) ? wordIDs1.get(temp) : wordIDs1.get(CombineParameters.UNKNOWN);
			
	 }
	
	public int getWordID2(CoreLabel token) {
		String temp = wordPreprocess(token, parameters);
		return wordIDs2.containsKey(temp) ? wordIDs2.get(temp) : wordIDs2.get(CombineParameters.UNKNOWN);
			
	 }
	
	public int getPosID(String s) {
	      return posIDs.containsKey(s) ? posIDs.get(s) : posIDs.get(CombineParameters.UNKNOWN);
	  }
	
	public int getPreSuffixID(String s) {
		return presuffixIDs.containsKey(s) ? presuffixIDs.get(s) : presuffixIDs.get(CombineParameters.UNKNOWN);
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

		return brownIDs.containsKey(s) ? brownIDs.get(s) : brownIDs.get(CombineParameters.UNKNOWN);
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
			return synsetIDs.get(CombineParameters.UNKNOWN);
		else
			return synsetIDs.containsKey(s) ? synsetIDs.get(s) : synsetIDs.get(CombineParameters.UNKNOWN);
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
			return hyperIDs.get(CombineParameters.UNKNOWN);
		else
			return hyperIDs.containsKey(s) ? hyperIDs.get(s) : hyperIDs.get(CombineParameters.UNKNOWN);
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
			return dictIDs.get(CombineParameters.UNKNOWN);
		else
			return dictIDs.containsKey(s) ? dictIDs.get(s) : dictIDs.get(CombineParameters.UNKNOWN);
	}
	
	public int getEntityTypeID(Entity entity) {
		return entitytypeIDs.containsKey(entity.type) ? entitytypeIDs.get(entity.type) : entitytypeIDs.get(CombineParameters.UNKNOWN);
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
				  Entity chem = new Entity(null, CombineParameters.CHEMICAL, current.beginPosition(), 
						  current.word(), null);
				  chem.start = idx;
				  chem.end = idx;
				  prediction.entities.add(chem);
			} else if(curTran==2) {// new disease
				CoreLabel current = tokens.get(idx);
				  Entity disease = new Entity(null, CombineParameters.DISEASE, current.beginPosition(), 
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
							RelationEntity relationEntity = new RelationEntity(CombineParameters.RELATION, former, latter);
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
					RelationEntity relationEntity = new RelationEntity(CombineParameters.RELATION, former, latter);
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
	public static String wordPreprocess(CoreLabel token, CombineParameters parameters) {
		if(parameters.wordPreprocess == 0) {
			return normalize_to_lowerwithdigit(token.word());
		} else if(parameters.wordPreprocess == 1)
			return token.word(); 
		else if(parameters.wordPreprocess==2) {
			return token.lemma().toLowerCase();
		} else if(parameters.wordPreprocess==3) {
			return Combine.pipe(token.lemma().toLowerCase());
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


	public double[][] getE() {
		return E;
	}

	public double[][] getEg2E() {
		// TODO Auto-generated method stub
		return eg2E;
	}
	


	public List<String> getKnownWords1() {
		// TODO Auto-generated method stub
		return knownWords1;
	}

	public List<String> getKnownWords2() {
		// TODO Auto-generated method stub
		return knownWords2;
	}

}



