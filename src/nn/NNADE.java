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

public class NNADE implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6814631308712316743L;
	public NN nn;
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
	public List<String> knownRelationDict;
	public TIntArrayList knownPosition;
	
	// key-word, value-word ID (the row id of the embedding matrix)
	public TObjectIntHashMap<String> wordIDs;
	public TObjectIntHashMap<String> posIDs;
	public TObjectIntHashMap<String> presuffixIDs;
	public TObjectIntHashMap<String> brownIDs;
	public TObjectIntHashMap<String> synsetIDs;
	public TObjectIntHashMap<String> hyperIDs;
	public TObjectIntHashMap<String> dictIDs;
	public TObjectIntHashMap<String> entitytypeIDs;
	public TObjectIntHashMap<String> relationDictIDs;
	public TIntIntHashMap positionIDs;
	
	// only used when loading external embeddings
	public TObjectIntHashMap<String> embedID;
	public double[][] embeddings;

	// the embedding matrix, embedding numbers x embeddingSize
	public double[][] E;
	public double[][] eg2E;
	
	// Store the high-frequency token-position
	public TIntArrayList preComputed;
	
	public boolean debug;
	
	public BrownCluster brownCluster;
	transient public IDictionary wordnet;
	
	public static final int UNKNOWN_POSITION = 101;
	
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
		
		// load relevant tools
		Tool tool = new Tool();
		tool.tokenizer = new Tokenizer(true, ' ');	
		tool.tagger = new MaxentTagger(PropertiesUtils.getString(properties, "pos_tagger", ""));
		tool.morphology = new Morphology();
		BrownCluster brown = new BrownCluster(PropertiesUtils.getString(properties, "brown_cluster_path", ""), 100);
		IDictionary dict = new edu.mit.jwi.Dictionary(new URL("file", null, PropertiesUtils.getString(properties, "wordnet_dict", "")));
		dict.open();
		
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
			
			NNADE nnade = new NNADE(parameters);
			nnade.debug = Boolean.parseBoolean(args[1]);
			nnade.brownCluster = brown;
			nnade.wordnet = dict;
			// save model for each group
			System.out.println(Parameters.SEPARATOR+" group "+i);
			BestPerformance best = nnade.trainAndTest(trainAb, testAb,modelFile+i, embedFile, tool);
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
			String embedFile, Tool tool) 
		throws Exception {
		// generate alphabet
		List<String> word = new ArrayList<>();
		List<String> pos = new ArrayList<>();	
		List<String> presuffix = new ArrayList<>();
		List<String> brown = new ArrayList<>();
		List<String> synset = new ArrayList<>();
		List<String> hyper = new ArrayList<>();
		
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = prepareNLPInfo(tool, sentence);
				
				for(CoreLabel token:tokens) {
					
					word.add(wordPreprocess(token, parameters));
					pos.add(token.tag());
					presuffix.add(getPrefix(token));
					presuffix.add(getSuffix(token));
					brown.add(getBrown(token));
					
					String id = getSynset(token);
					if(id!=null)
						synset.add(id);
					
					String hyperID = getHyper(token);
					if(hyperID!=null)
						hyper.add(hyperID);
				}
				
				
				for(Entity entity:sentence.entities) {
					String id = getSynset(entity.text);
					if(id!=null)
						synset.add(id);
					
					String hyperID = getHyper(entity.text);
					if(hyperID!=null)
						hyper.add(hyperID);
					
				}
			}
		}
	    	   
	    // Generate "dictionaries," possibly with frequency cutoff
	    knownWords = Util.generateDict(word, parameters.wordCutOff);
	    knownWords.add(0, Parameters.UNKNOWN);
	    knownWords.add(1, Parameters.PADDING);
	    knownPos = Util.generateDict(pos, 1);
	    knownPos.add(0, Parameters.UNKNOWN);
	    knownPos.add(1, Parameters.PADDING);
	    knownPreSuffix = Util.generateDict(presuffix, 1);
	    knownPreSuffix.add(0, Parameters.UNKNOWN);
	    knownPreSuffix.add(1, Parameters.PADDING);
	    knownBrown = Util.generateDict(brown, 1);
	    knownBrown.add(0, Parameters.UNKNOWN);
	    knownBrown.add(1, Parameters.PADDING);
	    knownSynSet = Util.generateDict(synset, 1);
	    knownSynSet.add(0, Parameters.UNKNOWN);
	    knownSynSet.add(1, Parameters.PADDING);
	    knownHyper = Util.generateDict(hyper, 1);
	    knownHyper.add(0, Parameters.UNKNOWN);
	    knownHyper.add(1, Parameters.PADDING);
	    
	    knownDict = new ArrayList<>();
	    knownDict.add(Parameters.UNKNOWN);
	    knownDict.add(Parameters.PADDING);
	    knownDict.add("disease");
	    knownDict.add("drug");
	    
	    knownEntityType = new ArrayList<>();
	    knownEntityType.add(Parameters.UNKNOWN);
	    knownEntityType.add("Disease");
	    knownEntityType.add("Chemical");
	    
	    knownRelationDict  = new ArrayList<>();
	    knownRelationDict.add(Parameters.UNKNOWN);
	    knownRelationDict.add("ade");
	    
	    knownPosition = new TIntArrayList();
	    knownPosition.add(UNKNOWN_POSITION); // the position we didn't see before
	    knownPosition.add(0); // the position in the entity
	    for(int i=1;i<=UNKNOWN_POSITION-1;i++) {
	    	knownPosition.add(i);
	    	knownPosition.add(-i);
	    }
	    
	    // Generate word id which can be used in the embedding matrix
	    wordIDs = new TObjectIntHashMap<String>();
	    posIDs = new TObjectIntHashMap<>();
	    presuffixIDs = new TObjectIntHashMap<>();
	    brownIDs = new TObjectIntHashMap<>();
	    synsetIDs = new TObjectIntHashMap<>();
	    hyperIDs = new TObjectIntHashMap<>();
	    dictIDs = new TObjectIntHashMap<>();
	    entitytypeIDs = new TObjectIntHashMap<>();
	    relationDictIDs = new TObjectIntHashMap<>();
	    positionIDs = new TIntIntHashMap();
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
	    for(String temp:knownRelationDict)
	    	relationDictIDs.put(temp, (m++));
	    for(int i=0;i<knownPosition.size();i++)
	    	positionIDs.put(knownPosition.get(i), (m++));

	    System.out.println("#Word: " + knownWords.size());
	    System.out.println("#POS: " + knownPos.size());
	    System.out.println("#PreSuffix: " + knownPreSuffix.size());
	    System.out.println("#Brown: " + knownBrown.size());
	    System.out.println("#Synset: " + knownSynSet.size());
	    System.out.println("#Hyper: " + knownHyper.size());
	    System.out.println("#Dict: " + knownDict.size());
	    System.out.println("#Entity type: "+knownEntityType.size());
	    System.out.println("#Relation Dict: "+knownRelationDict.size());
	    System.out.println("#Position: "+knownPosition.size());
		
	    // 2 represents the class of disease and drug of the dictionaries, they always at the end of E.
		E = new double[knownWords.size()+knownPos.size()+knownPreSuffix.size()+knownBrown.size()+
		                          knownSynSet.size()+knownHyper.size()+knownDict.size()+knownEntityType.size()
		                          +knownRelationDict.size()+knownPosition.size()][parameters.embeddingSize];
		eg2E = new double[E.length][E[0].length];
		Random random = new Random(System.currentTimeMillis());
		if(embedFile!=null && !embedFile.isEmpty()) { // try to load off-the-shelf embeddings
		    embedID = new TObjectIntHashMap<String>();
		    BufferedReader input = null;
		    
			  input = IOUtils.readerFromString(embedFile);
			  List<String> lines = new ArrayList<String>();
			  for (String s; (s = input.readLine()) != null; ) {
			    lines.add(s);
			  }
			
			  
			  String[] splits = lines.get(0).split("\\s+");
			  
			  int nWords = Integer.parseInt(splits[0]);
			  int dim = Integer.parseInt(splits[1]);
			  lines.remove(0);
			  embeddings = new double[nWords][dim];
			
			  
			  for (int i = 0; i < lines.size(); ++i) {
			    splits = lines.get(i).split("\\s+");
			    embedID.put(splits[0], i);
			    for (int j = 0; j < dim; ++j)
			      embeddings[i][j] = Double.parseDouble(splits[j + 1]);
			  }
			  
			  // using loaded embeddings to initial E
			  
			  for (int i = 0; i < E.length; ++i) {
			    int index = -1;
			    if (i < knownWords.size()) {
			      String str = knownWords.get(i);
			      //NOTE: exact match first, and then try lower case..
			      if (embedID.containsKey(str)) index = embedID.get(str);
			      else if (embedID.containsKey(str.toLowerCase())) index = embedID.get(str.toLowerCase());
			    }
			
			    if (index >= 0) {
			      for (int j = 0; j < E[0].length; ++j)
			        E[i][j] = embeddings[index][j];
			    } else {
			      for (int j = 0; j < E[0].length; ++j)
			        E[i][j] = random.nextDouble() * parameters.initRange * 2 - parameters.initRange;
			    }
			  }
			
		} else { // initialize E randomly
			System.out.println("No Embedding File, so initialize E randomly!");
			for(int i=0;i<E.length;i++) {
				for(int j=0;j<E[0].length;j++) {
					E[i][j] = random.nextDouble() * parameters.initRange * 2 - parameters.initRange;
				}
			}
		}

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
		nn  = new NN(parameters, this, preComputed, exampleEntity.get(0));
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
			
			GradientKeeper keeper = nn.process(examples);
			
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
        		if(parameters.beamSize > 0) {
	        		Beam beam = new Beam(parameters.beamSize);
	        		predicted = decodeWithBeam(tokens, tool, beam);
        		} else {
        			predicted = decode(tokens, stat, tool);
        		}

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
        System.out.println(Parameters.SEPARATOR);

        	
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
					int[] defalutValues = {0,0,0,0,0,0};
					TIntArrayList goldLabel = new TIntArrayList(defalutValues);
					CoreLabel token = tokens.get(idx);
					
					int index = Util.isInsideAGoldEntityAndReturnIt(token.beginPosition(), token.endPosition(), entities);
					
					if(index == -1) {
						// other
						goldLabel.set(0, 1);
					} else {
						Entity gold = entities.get(index);
						if(Util.isFirstWordOfEntity(gold, token)) {
							if(gold.type.equals(Parameters.CHEMICAL)) {
								// new chemical
								goldLabel.set(1, 1);
							} else {
								// new disease
								goldLabel.set(2, 1);
							}
						} else {
							// append
							goldLabel.set(3, 1);
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
						int[] defalutValues = {0,0,0,0,0,0};
						TIntArrayList goldLabel = new TIntArrayList(defalutValues);
						
						RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
						if(sentence.relaitons.contains(tempRelation)) {
							// connect
							goldLabel.set(5, 1);
						} else {
							// not connect
							goldLabel.set(4, 1);
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
			//example.featureIdx.add(getBrownID(getBrown(tokens.get(idx))));
			
			// context brown
			/*for(int i=0;i<2;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getBrownID(getBrown(tokens.get(idxBefore))));
				} else {
					example.featureIdx.add(getBrownID(Parameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getBrownID(getBrown(tokens.get(idxAfter))));
				} else {
					example.featureIdx.add(getBrownID(Parameters.PADDING));
				}
			}*/
			
			// current synset
			//example.featureIdx.add(getSynsetID(getSynset(tokens.get(idx))));
			
			// context synset
			/*for(int i=0;i<2;i++) {
				int idxBefore = idx-1-i;
				if(idxBefore>=0) {
					example.featureIdx.add(getSynsetID(getSynset(tokens.get(idxBefore))));
				} else {
					example.featureIdx.add(getSynsetID(Parameters.PADDING));
				}
			}
			for(int i=0;i<2;i++) {
				int idxAfter = idx+1+i;
				if(idxAfter<=tokens.size()-1) {
					example.featureIdx.add(getSynsetID(getSynset(tokens.get(idxAfter))));
				} else {
					example.featureIdx.add(getSynsetID(Parameters.PADDING));
				}
			}*/
			
			// current hyper
			//example.featureIdx.add(getHyperID(getHyper(tokens.get(idx))));
			
			// current dict
			//example.featureIdx.add(getDictID(getDict(tokens.get(idx), tool)));
			// context dict
			/*for(int i=0;i<2;i++) {
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
			}*/
			
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
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			
			// entity wordnet
			/*example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);
			example.featureIdx.add(-1);*/
			
					
			
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
			//example.featureIdx.add(-1);
			
			// context brown
			/*for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}*/
			
			// current synset
			//example.featureIdx.add(-1);
			
			// context synset
			/*for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}*/
			
			// current hyper
			//example.featureIdx.add(-1);
						
			// current dict
			//example.featureIdx.add(-1);
			// context dict
			/*for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}
			for(int i=0;i<2;i++) {
				example.featureIdx.add(-1);
			}*/
			
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
			example.featureIdx.add(getEntityTypeID(former));
			example.featureIdx.add(getEntityTypeID(latter));
			
			// entity wordnet
			/*example.featureIdx.add(getSynsetID(getSynset(former.text)));
			example.featureIdx.add(getSynsetID(getSynset(latter.text)));
			example.featureIdx.add(getHyperID(getHyper(former.text)));
			example.featureIdx.add(getHyperID(getHyper(latter.text)));*/
			
									
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
				example.positionIdxFormer.add(getPositionID(UNKNOWN_POSITION));
				example.positionIdxLatter.add(getPositionID(UNKNOWN_POSITION));
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
	
	public String getBrown(CoreLabel token) {
		return brownCluster.getPrefix(token.lemma());
	}
	
	public int getBrownID(String s) {

		return brownIDs.containsKey(s) ? brownIDs.get(s) : brownIDs.get(Parameters.UNKNOWN);
	}
	
	public String getSynset(CoreLabel token) {
		ISynset synset = WordNetUtil.getMostSynset(wordnet, token.lemma(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostSynset(wordnet, token.lemma(), POS.ADJECTIVE);
			if(synset!= null)
				return String.valueOf(synset.getID().getOffset());
			else
				return null;
		}
	}
	
	public String getSynset(String s) {
		ISynset synset = WordNetUtil.getMostSynset(wordnet, s.toLowerCase(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostSynset(wordnet, s.toLowerCase(), POS.ADJECTIVE);
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
	
	public String getHyper(CoreLabel token) {
		ISynset synset = WordNetUtil.getMostHypernym(wordnet, token.lemma(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostHypernym(wordnet, token.lemma(), POS.ADJECTIVE);
			if(synset != null)
				return String.valueOf(synset.getID().getOffset());
			else
				return null;
		}
	}
	
	public String getHyper(String s) {
		ISynset synset = WordNetUtil.getMostHypernym(wordnet, s.toLowerCase(), POS.NOUN);
		if(synset!= null) {
			return String.valueOf(synset.getID().getOffset());
		} else {
			synset = WordNetUtil.getMostHypernym(wordnet, s.toLowerCase(), POS.ADJECTIVE);
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
		
		
		return null;	
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
	
	public int getRelationDictID(String s) {
		if(s==null)
			return relationDictIDs.get(Parameters.UNKNOWN);
		else
			return relationDictIDs.containsKey(s) ? relationDictIDs.get(s) : relationDictIDs.get(Parameters.UNKNOWN);
	}
	
	public int getPositionID(int position) {
		return positionIDs.contains(position) ? positionIDs.get(position) : positionIDs.get(UNKNOWN_POSITION);
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
	public ADESentence decodeWithBeam(List<CoreLabel> tokens, Tool tool, Beam beam) throws Exception {
		
		for(int idx=0;idx<tokens.size();idx++) {
			List<Prediction> buf = new ArrayList<>();
			// prepare the input for NN
			Example ex = getExampleFeatures(tokens, idx, false, null, null, tool, null);
			double[] scores1 = nn.computeScores(ex);
			if(parameters.beamMeanSoftmax)
				softmaxScores(scores1);
			
			if(beam.items.size()==0) { // generate predictions without copy
				for (int i = 0; i < parameters.outputSize-2; ++i) {
		            Prediction prediction = new Prediction(parameters.beamMeanSoftmax);
		            prediction.addLabel(i, scores1[i]);
		            buf.add(prediction);

		        }
			} else { // generate predictions with copying items in the beam
				for(int k=0;k<beam.items.size();k++) {
			        for (int i = 0; i < parameters.outputSize-2; ++i) {
			            Prediction prediction = new Prediction(parameters.beamMeanSoftmax);
			            prediction.copy((Prediction)beam.items.get(k));
			            prediction.addLabel(i, scores1[i]);
			            buf.add(prediction);

			        }
				}
			}
			beam.kbest(buf);
			buf.clear();
			// generate entities based on the latest label
			for(int beamIdx=0;beamIdx<beam.items.size();beamIdx++) {
				Prediction prediction = beam.items.get(beamIdx);
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
			}
			
			// begin to predict relations
			for(int beamIdx=0;beamIdx<beam.items.size();beamIdx++) {
				Prediction beamPrediction = beam.items.get(beamIdx);
				int lastTran = beamPrediction.labels.size()>=2 ? beamPrediction.labels.get(beamPrediction.labels.size()-2) : -1;
				int curTran = beamPrediction.labels.get(beamPrediction.labels.size()-1);
				// judge whether to generate relations
				if((lastTran==1 && curTran==1) || (lastTran==1 && curTran==2) || (lastTran==1 && curTran==0)
						|| (lastTran==2 && curTran==0) || (lastTran==2 && curTran==1) || (lastTran==2 && curTran==2)
						|| (lastTran==3 && curTran==0) || (lastTran==3 && curTran==1) || (lastTran==3 && curTran==2)
				) { 
					if((lastTran==3 && checkWrongState(beamPrediction)) || lastTran==1 || lastTran==2) {
						// if curTran 1 or 2, the last entities should not be considered
						int latterIdx = (curTran==1 || curTran==2) ? beamPrediction.entities.size()-2:beamPrediction.entities.size()-1;
						Entity latter = beamPrediction.entities.get(latterIdx);
						List<Prediction> buf1 = new ArrayList<>();
						buf1.add(beamPrediction);
						List<Prediction> buf2 = new ArrayList<>();
						for(int j=0;j<latterIdx;j++) {
							Entity former = beamPrediction.entities.get(j);
							Example relationExample = getExampleFeatures(tokens, idx, true, former, latter, tool, beamPrediction.entities);
							double[] scores2 = nn.computeScores(relationExample);
							if(parameters.beamMeanSoftmax)
								softmaxScores(scores2);
							for(int k=0;k<buf1.size();k++) {
						        for (int i = parameters.outputSize-2; i < parameters.outputSize; ++i) {
						            Prediction prediction = new Prediction(parameters.beamMeanSoftmax);
						            prediction.copy(buf1.get(k));
						            prediction.addLabel(i, scores2[i]);
						            buf2.add(prediction);
						            // generate relations based on the latest label
						            curTran = prediction.labels.get(prediction.labels.size()-1);
						        	if(curTran == 5) { // connect
										RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
										prediction.relations.add(relationEntity);
						        	}
						        }
						        
							}
							buf1.clear();
							buf1.addAll(buf2);
							buf2.clear();
						}
						buf.addAll(buf1);
					} else
						buf.add(beamPrediction);
					
				} else
					buf.add(beamPrediction);
				
			}
			
			beam.kbest(buf);
			buf.clear();

		}
		
		// when at the end of sentence, judge relation ignoring lastTran
		List<Prediction> buf = new ArrayList<>();
		for(int beamIdx=0;beamIdx<beam.items.size();beamIdx++) {
			Prediction beamPrediction = beam.items.get(beamIdx);
			int curTran = beamPrediction.labels.get(beamPrediction.labels.size()-1);
			if((curTran==3 && checkWrongState(beamPrediction)) || curTran==1 || curTran==2) {
				int latterIdx = beamPrediction.entities.size()-1;
				Entity latter = beamPrediction.entities.get(latterIdx);
				List<Prediction> buf1 = new ArrayList<>();
				buf1.add(beamPrediction);
				List<Prediction> buf2 = new ArrayList<>();
				for(int j=0;j<latterIdx;j++) {
					Entity former = beamPrediction.entities.get(j);
					Example relationExample = getExampleFeatures(tokens, tokens.size()-1, true, former, latter, tool, beamPrediction.entities);
					double[] scores2 = nn.computeScores(relationExample);
					if(parameters.beamMeanSoftmax)
						softmaxScores(scores2);
					for(int k=0;k<buf1.size();k++) {
				        for (int i = parameters.outputSize-2; i < parameters.outputSize; ++i) {
				            Prediction prediction = new Prediction(parameters.beamMeanSoftmax);
				            prediction.copy(buf1.get(k));
				            prediction.addLabel(i, scores2[i]);
				            buf2.add(prediction);
				            // generate relations based on the latest label
				            curTran = prediction.labels.get(prediction.labels.size()-1);
				        	if(curTran == 5) { // connect
								RelationEntity relationEntity = new RelationEntity(Parameters.RELATION, former, latter);
								prediction.relations.add(relationEntity);
				        	}
				        }
				        
					}
					buf1.clear();
					buf1.addAll(buf2);
					buf2.clear();
				}
				buf.addAll(buf1);
			} else
				buf.add(beamPrediction);
		}
		beam.kbest(buf);
		buf.clear();
		
		
		// Prediction to ADESentence
		Prediction best = beam.items.get(0);
		ADESentence predicted = new ADESentence();
		predicted.entities.addAll(best.entities);
		predicted.relaitons.addAll(best.relations);
		
		return predicted;
	}
	
	public ADESentence decode(List<CoreLabel> tokens, DecodeStatistic stat, Tool tool) throws Exception {
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
	
	// Given a raw sentence, output the prediction
	// old version, may have some problems because 
	// didn't handle lastTran==-1, may lost entities
	// didn't handle the sentence end, may lost relations
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
				|| (lastTran==3 && curTran==1) || (lastTran==4 && curTran==1) || (lastTran==5 && curTran==1))
			{ // new chemical
				CoreLabel current = tokens.get(idx);
				  Entity chem = new Entity(null, Parameters.CHEMICAL, current.beginPosition(), 
						  current.word(), null);
				  chem.start = idx;
				  chem.end = idx;
				  tempEntities.add(chem);
				  newed = true;
			} else if((lastTran==0 && curTran==2) || (lastTran==1 && curTran==2) || (lastTran==2 && curTran==2)
				|| (lastTran==3 && curTran==2) || (lastTran==4 && curTran==2) || (lastTran==5 && curTran==2))
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
			} else { // last or current = -1 but not wrong status
				
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
		if(parameters.wordPreprocess == 1)
			return token.word(); 
		else if(parameters.wordPreprocess==2) {
			return token.lemma().toLowerCase();
		} else if(parameters.wordPreprocess==3) {
			return NNADE.pipe(token.lemma().toLowerCase());
		} else {
			return token.word().toLowerCase();
		}

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

}

class DecodeStatistic {
	public int wrong; // wrong transition
	public int total; // total transition
	
	public int ctPredictEntity = 0;
	public int ctTrueEntity = 0;
	public int ctCorrectEntity = 0;
	public int ctPredictRelation = 0;
	public int ctTrueRelation = 0;
	public int ctCorrectRelation = 0;
	
	public double getWrongRate() {
		return wrong*1.0/total;
	}
	
	public double getEntityPrecision() {
		return Evaluater.getPrecisionV2(ctCorrectEntity, ctPredictEntity);
	}
	
	public double getEntityRecall() {
		return Evaluater.getRecallV2(ctCorrectEntity, ctTrueEntity);
	}
	
	public double getEntityF1() {
		return Evaluater.getFMeasure(getEntityPrecision(), getEntityRecall(), 1);
	}
	
	public double getRelationPrecision() {
		return Evaluater.getPrecisionV2(ctCorrectRelation, ctPredictRelation);
	}
	
	public double getRelationRecall() {
		return Evaluater.getRecallV2(ctCorrectRelation, ctTrueRelation);
	}
	
	public double getRelationF1() {
		return Evaluater.getFMeasure(getRelationPrecision(), getRelationRecall(), 1);
	}
}

class BestPerformance {
	public double pEntity = -1;
	public double rEntity= -1;
	public double f1Entity= -1;
	public double pRelation= -1;
	public double rRelation= -1;
	public double f1Relation= -1;
	
	
}
