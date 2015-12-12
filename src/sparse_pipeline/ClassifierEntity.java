package sparse_pipeline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import cc.mallet.classify.Classifier;
import cn.fox.biomedical.Dictionary;
import cn.fox.machine_learning.BrownCluster;
import cn.fox.stanford.Tokenizer;
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

public class ClassifierEntity extends Father implements Serializable {
	
	public Parameters parameters;
	SparseLayer sparse;
	
		
	public TObjectIntHashMap<String> featureIDs;
	public boolean freezeAlphabet;

	public static void main(String[] args) throws Exception {
		FileInputStream fis = new FileInputStream(args[0]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
		boolean debug = Boolean.parseBoolean(args[1]);
		
		Parameters parameters = new Parameters(properties);
		parameters.printParameters();
		
		File fAbstractDir = new File(PropertiesUtils.getString(properties, "corpusDir", ""));
		File groupFile = new File(PropertiesUtils.getString(properties, "groupFile", ""));
		String modelFile = args[2];
		
		
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
			
			ClassifierEntity classifier = new ClassifierEntity(parameters);
			
			System.out.println(Parameters.SEPARATOR+" group "+i);
			BestPerformance best = classifier.trainAndTest(trainAb, testAb,modelFile+i, tool, debug);
			bestAll.add(best);
			
		}
		
		// macro performance
		double macroP_Entity = 0;
		double macroR_Entity = 0;
		double macroF1_Entity = 0;
		double fold = bestAll.size();
		for(BestPerformance best:bestAll) {
			macroP_Entity += best.pEntity/fold; 
			macroR_Entity += best.rEntity/fold;
		}
		macroF1_Entity = Evaluater.getFMeasure(macroP_Entity, macroR_Entity, 1);
		
		System.out.println("macro entity precision\t"+macroP_Entity);
        System.out.println("macro entity recall\t"+macroR_Entity);
        System.out.println("macro entity f1\t"+macroF1_Entity);

	}
	
	public BestPerformance trainAndTest(List<Abstract> trainAbs, List<Abstract> testAbs, String modelFile, 
			Tool tool, boolean debug) 
		throws Exception {
		
	    
		// generate training examples
		// generate alphabet simultaneously
		featureIDs = new TObjectIntHashMap<>();
		List<Example> exampleEntity = generateEntityTrainExamples(trainAbs, tool);
		freezeAlphabet = true;
		System.out.println("Total sparse feature number: "+featureIDs.size());
		// new a NN and initialize its weight
		sparse  = new SparseLayer(parameters, this, featureIDs.size(), 4, debug, false);
		
		// train iteration
		long startTime = System.currentTimeMillis();
		BestPerformance best = new BestPerformance();
		
		for (int iter = 0; iter < parameters.maxIter; ++iter) {
						
			// mini-batch
			int batchSizeEntity = (int)(exampleEntity.size()*parameters.batchEntityPercent);
			if(batchSizeEntity == 0)
				batchSizeEntity++;
			List<Example> batchExampleEntity = Util.getRandomSubList(exampleEntity, batchSizeEntity);
			
			GradientKeeper keeper = sparse.process(batchExampleEntity);
			
			//sparse.checkGradients(keeper, batchExampleEntity);
			sparse.updateWeights(keeper);
			
			
			if (iter>0 && iter % parameters.evalPerIter == 0) {
				evaluate(tool, testAbs, modelFile, best);
			}			
		}
		
		evaluate(tool, testAbs, modelFile, best);
		
		return best;
	}
	
	public void evaluate(Tool tool, List<Abstract> testAbs, String modelFile, BestPerformance best)
			throws Exception {
		

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
        System.out.println(Parameters.SEPARATOR);

        	
        if ((f1Entity > best.f1Entity)) {
          System.out.printf("Current Exceeds the best! Saving model file %s\n", modelFile);
          best.pEntity = pEntity;
          best.rEntity = rEntity;
          best.f1Entity = f1Entity;
          ObjectSerializer.writeObjectToFile(this, modelFile);
        }
        
	}
	
	public ADESentence decode(List<CoreLabel> tokens, Tool tool) throws Exception {
		Prediction prediction = new Prediction();
		for(int idx=0;idx<tokens.size();idx++) {
			// prepare the input for NN
			Example ex = getExampleFeatures(tokens, idx, false, tool);
			int transition = sparse.giveTheBestChoice(ex);
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

			
		}
		
				
		// Prediction to ADESentence
		ADESentence predicted = new ADESentence();
		predicted.entities.addAll(prediction.entities);
		predicted.relaitons.addAll(prediction.relations);
		
		return predicted;
	}
	
	public boolean checkWrongState(Prediction prediction) {
		int position045 = -1;
		int positionOf1or2 = -1;
		for(int j=prediction.labels.size()-2;j>=0;j--) {
			if(prediction.labels.get(j)==1 || prediction.labels.get(j)==2)
				positionOf1or2 = j;
			else if(prediction.labels.get(j)==0)
				position045 = j;
			 
			
			if(position045!=-1 && positionOf1or2!=-1)
				break;
		}
		
		if(position045 < positionOf1or2) 
			return true;
		else
			return false;
	}
	
	public int getFeatureId(String name) {
		if(freezeAlphabet) {
			return featureIDs.contains(name) ? featureIDs.get(name) : -1;
		} else {
			if(featureIDs.contains(name)) {
				return featureIDs.get(name);
			} else {
				featureIDs.put(name, featureIDs.size());
				return featureIDs.get(name);
			}
		}
	}
	
	public Example getExampleFeatures(List<CoreLabel> tokens, int idx, boolean bRelation,
			Tool tool) throws Exception {
		Example example = new Example(bRelation);
		
		// current word
		example.featureIdx.add(getFeatureId("WD#"+wordPreprocess(tokens.get(idx), parameters)));
		
		// words before the current word, but in the window
		for(int i=0;i<2;i++) {
			int idxBefore = idx-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("WDB#"+wordPreprocess(tokens.get(idxBefore), parameters)));
			} else {
				example.featureIdx.add(getFeatureId("WDB#"+Parameters.PADDING));
			}
		}
		// words after the current word, but in the window
		for(int i=0;i<2;i++) {
			int idxAfter = idx+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("WDA#"+wordPreprocess(tokens.get(idxAfter), parameters)));
			} else {
				example.featureIdx.add(getFeatureId("WDA#"+Parameters.PADDING));
			}
		}
		
		// current pos
		example.featureIdx.add(getFeatureId("POS#"+tokens.get(idx).tag()));
		
		// context pos
		for(int i=0;i<1;i++) {
			int idxBefore = idx-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("POSB#"+tokens.get(idxBefore).tag()));
			} else {
				example.featureIdx.add(getFeatureId("POSB#"+Parameters.PADDING));
			}
		}
		for(int i=0;i<1;i++) {
			int idxAfter = idx+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("POSA#"+tokens.get(idxAfter).tag()));
			} else {
				example.featureIdx.add(getFeatureId("POSA#"+Parameters.PADDING));
			}
		}
		
		// current prefix and suffix
		example.featureIdx.add(getFeatureId("PREF#"+getPrefix(tokens.get(idx))));
		example.featureIdx.add(getFeatureId("SUF#"+getSuffix(tokens.get(idx))));
		
		// context prefix and suffix
		for(int i=0;i<2;i++) {
			int idxBefore = idx-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("PREFB#"+getPrefix(tokens.get(idxBefore))));
				example.featureIdx.add(getFeatureId("SUFB#"+getSuffix(tokens.get(idxBefore))));
			} else {
				example.featureIdx.add(getFeatureId("PREFB#"+Parameters.PADDING));
				example.featureIdx.add(getFeatureId("SUFB#"+Parameters.PADDING));
			}
		}
		for(int i=0;i<2;i++) {
			int idxAfter = idx+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("PREFA#"+getPrefix(tokens.get(idxAfter))));
				example.featureIdx.add(getFeatureId("SUFA#"+getSuffix(tokens.get(idxAfter))));
			} else {
				example.featureIdx.add(getFeatureId("PREFA#"+Parameters.PADDING));
				example.featureIdx.add(getFeatureId("SUFA#"+Parameters.PADDING));
			}
		}
		
		// current brown
		example.featureIdx.add(getFeatureId("BRN#"+getBrown(tokens.get(idx), tool)));
		
		// context brown
		for(int i=0;i<2;i++) {
			int idxBefore = idx-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("BRNB#"+getBrown(tokens.get(idxBefore), tool)));
			} else {
				example.featureIdx.add(getFeatureId("BRNB#"+Parameters.PADDING));
			}
		}
		for(int i=0;i<2;i++) {
			int idxAfter = idx+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("BRNA#"+getBrown(tokens.get(idxAfter), tool)));
			} else {
				example.featureIdx.add(getFeatureId("BRNA#"+Parameters.PADDING));
			}
		}
		
		// current synset
		example.featureIdx.add(getFeatureId("SYN#"+getSynset(tokens.get(idx), tool)));
		
		// context synset
		for(int i=0;i<2;i++) {
			int idxBefore = idx-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("SYNB#"+getSynset(tokens.get(idxBefore), tool)));
			} else {
				example.featureIdx.add(getFeatureId("SYNB#"+Parameters.PADDING));
			}
		}
		for(int i=0;i<2;i++) {
			int idxAfter = idx+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("SYNA#"+getSynset(tokens.get(idxAfter), tool)));
			} else {
				example.featureIdx.add(getFeatureId("SYNA#"+Parameters.PADDING));
			}
		}
		
		// current hyper
		example.featureIdx.add(getFeatureId("HYP#"+getHyper(tokens.get(idx), tool)));
		
		// current dict
		example.featureIdx.add(getFeatureId("DIC#"+getDict(tokens.get(idx), tool)));
		// context dict
		for(int i=0;i<2;i++) {
			int idxBefore = idx-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("DICB#"+getDict(tokens.get(idxBefore), tool)));
			} else {
				example.featureIdx.add(getFeatureId("DICB#"+Parameters.PADDING));
			}
		}
		for(int i=0;i<2;i++) {
			int idxAfter = idx+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("DICA#"+getDict(tokens.get(idxAfter), tool)));
			} else {
				example.featureIdx.add(getFeatureId("DICA#"+Parameters.PADDING));
			}
		}
		
		


		return example;
	}
	
	public List<Example> generateEntityTrainExamples(List<Abstract> trainAbs, Tool tool)
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
					Example example = getExampleFeatures(tokens, idx, false, tool);
					double[] goldLabel = {0,0,0,0};
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
					
										
				}
				
				
			}
		}
		
		return ret;
	}
	
	public static String getSynset(CoreLabel token, Tool tool) {
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
	
	public static String getSynset(String s, Tool tool) {
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
	
	public static String getHyper(CoreLabel token, Tool tool) {
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
	
	public static String getHyper(String s, Tool tool) {
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
	
	public String getBrown(CoreLabel token, Tool tool) {
		return tool.brownCluster.getPrefix(token.lemma());
	}
	
	public String getPrefix(CoreLabel token) {
		int len = token.lemma().length()>2 ? 2:token.lemma().length();
		return token.lemma().substring(0, len);
	}
	
	public String getSuffix(CoreLabel token) {
		int len = token.lemma().length()>2 ? 2:token.lemma().length();
		return token.lemma().substring(token.lemma().length()-len, token.lemma().length());
	}
	
	public static String wordPreprocess(CoreLabel token, Parameters parameters) {
		return token.lemma().toLowerCase();
	}
	
	public static List<CoreLabel> prepareNLPInfo(Tool tool, ADESentence sentence) {
		ArrayList<CoreLabel> tokens = tool.tokenizer.tokenize(sentence.offset, sentence.text);
		tool.tagger.tagCoreLabels(tokens);
		for(int i=0;i<tokens.size();i++)
			tool.morphology.stem(tokens.get(i));
		
		return tokens;
	}
	
	public ClassifierEntity(Parameters parameters) {
		
		this.parameters = parameters;
	}

	@Override
	public double[][] getE() {
		return null;
	}

	@Override
	public int getPaddingID() {
		
		return -1;
	}

	@Override
	public int getPositionID(int position) {
		return -1;
	}

	@Override
	public double[][] getEg2E() {
		return null;
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
	
	
}
