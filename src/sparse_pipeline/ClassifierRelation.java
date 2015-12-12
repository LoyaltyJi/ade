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
import java.util.Set;

import cn.fox.biomedical.Dictionary;
import cn.fox.machine_learning.BrownCluster;
import cn.fox.stanford.Tokenizer;
import cn.fox.utils.Evaluater;
import cn.fox.utils.ObjectSerializer;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;
import drug_side_effect_utils.Tool;
import edu.mit.jwi.IDictionary;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.PropertiesUtils;
import gnu.trove.TObjectIntHashMap;
import utils.ADESentence;
import utils.Abstract;

public class ClassifierRelation extends Father implements Serializable {
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
		String entityModelFile = args[3];
		
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
			
			ClassifierRelation classifierRelation = new ClassifierRelation(parameters);
			
			System.out.println(Parameters.SEPARATOR+" group "+i);
			BestPerformance best = classifierRelation.trainAndTest(trainAb, testAb,modelFile+i, tool, debug, entityModelFile+i);
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
	
	public Example getExampleFeatures(List<CoreLabel> tokens, boolean bRelation,
			Entity former, Entity latter, Tool tool) throws Exception {
		Example example = new Example(bRelation);
		
			
		// words before the former, but in the window
		for(int i=0;i<2;i++) {
			int idxBefore = former.start-1-i;
			if(idxBefore>=0) {
				example.featureIdx.add(getFeatureId("WDBF#"+ClassifierEntity.wordPreprocess(tokens.get(idxBefore), parameters)));
			} else {
				example.featureIdx.add(getFeatureId("WDBF#"+Parameters.PADDING));
			}
		}
		// words after the latter, but in the window
		for(int i=0;i<2;i++) {
			int idxAfter = latter.end+1+i;
			if(idxAfter<=tokens.size()-1) {
				example.featureIdx.add(getFeatureId("WDAL#"+ClassifierEntity.wordPreprocess(tokens.get(idxAfter), parameters)));
			} else {
				example.featureIdx.add(getFeatureId("WDAL#"+Parameters.PADDING));
			}
		}
		
		// words after the first entity
		for(int i=0;i<2;i++) {
			int idxAfter = former.end+1+i;
			if(idxAfter<=latter.start-1) {
				example.featureIdx.add(getFeatureId("WDAF#"+ClassifierEntity.wordPreprocess(tokens.get(idxAfter), parameters)));
			} else {
				example.featureIdx.add(getFeatureId("WDAF#"+Parameters.PADDING));
			}
			
		}
		// words before the second entity
		for(int i=0;i<2;i++) {
			int idxBefore = latter.start-1-i;
			if(idxBefore>=former.end+1) {
				example.featureIdx.add(getFeatureId("WDBL#"+ClassifierEntity.wordPreprocess(tokens.get(idxBefore), parameters)));
			} else {
				example.featureIdx.add(getFeatureId("WDBL#"+Parameters.PADDING));
			}
		}
		
		// entity type
		example.featureIdx.add(getFeatureId("ENT#"+former.type+"#"+latter.type));
		
		// entity wordnet
		example.featureIdx.add(getFeatureId("SYN#"+ClassifierEntity.getSynset(former.text, tool)+"#"+ClassifierEntity.getSynset(latter.text, tool)));
		example.featureIdx.add(getFeatureId("HYP#"+ClassifierEntity.getHyper(former.text, tool)+"#"+ClassifierEntity.getHyper(latter.text, tool)));
		
		// entity
		example.featureIdx.add(getFeatureId("EN#"+former.text.toLowerCase()+"#"+latter.text.toLowerCase()));
		
				
		return example;
	}
	
	public List<Example> generateRelationTrainExamples(List<Abstract> trainAbs, Tool tool)
			throws Exception {
		List<Example> ret = new ArrayList<>();
				
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = ClassifierEntity.prepareNLPInfo(tool, sentence);
				// resort the entities in the sentence
				List<Entity> entities = Util.resortEntity(sentence);
				// fill 'start' and 'end' of the entities
				Util.fillEntity(entities, tokens);
				
				// for each entity pair, we generate a relation example
				for(int i=0;i<entities.size();i++) {
					Entity latter = entities.get(i);
					for(int j=0;j<i;j++) {
						Entity former = entities.get(j);
						
						Example example = getExampleFeatures(tokens, true, former, latter, tool);
						double[] goldLabel = {0,0};
						
						RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
						if(sentence.relaitons.contains(tempRelation)) {
							// connect
							goldLabel[1] = 1;
						} else {
							// not connect
							goldLabel[0] = 1;
						}
						example.label = goldLabel;
						ret.add(example);
					}
				}
				
	
			}
		}
		
		return ret;
	}
	
	public BestPerformance trainAndTest(List<Abstract> trainAbs, List<Abstract> testAbs, String modelFile, 
			Tool tool, boolean debug, String entityModelFile) 
		throws Exception {
		ClassifierEntity classifierEntity = (ClassifierEntity)ObjectSerializer.readObjectFromFile(entityModelFile);
	    
		// generate training examples
		// generate alphabet simultaneously
		featureIDs = new TObjectIntHashMap<>();
		List<Example> exampleRelation = generateRelationTrainExamples(trainAbs, tool);
		freezeAlphabet = true;
		System.out.println("Total sparse feature number: "+featureIDs.size());
		// new a NN and initialize its weight
		sparse  = new SparseLayer(parameters, this, featureIDs.size(), 2, debug, false);
		
		// train iteration
		long startTime = System.currentTimeMillis();
		BestPerformance best = new BestPerformance();
		
		for (int iter = 0; iter < parameters.maxIter; ++iter) {
						
			// mini-batch
			int batchSizeRelation = (int)(exampleRelation.size()*parameters.batchRelationPercent);
			if(batchSizeRelation == 0)
				batchSizeRelation++;
			List<Example> batchExampleRelation = Util.getRandomSubList(exampleRelation, batchSizeRelation);
			
			GradientKeeper keeper = sparse.process(batchExampleRelation);
			
			//sparse.checkGradients(keeper, batchExampleEntity);
			sparse.updateWeights(keeper);
			
			
			if (iter>0 && iter % parameters.evalPerIter == 0) {
				evaluate(tool, testAbs, modelFile, best, classifierEntity);
			}			
		}
		
		evaluate(tool, testAbs, modelFile, best, classifierEntity);
		
		return best;
	}
	
	public void evaluate(Tool tool, List<Abstract> testAbs, String modelFile, BestPerformance best
			, ClassifierEntity classifierEntity)
			throws Exception {
		

		DecodeStatistic stat = new DecodeStatistic();
        for(Abstract testAb:testAbs) {
        	for(ADESentence gold:testAb.sentences) {
        		List<CoreLabel> tokens = ClassifierEntity.prepareNLPInfo(tool, gold);
        		ADESentence predicted = null;
        		predicted = decode(tokens, tool, classifierEntity);
        		
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
	
	public ADESentence decode(List<CoreLabel> tokens, Tool tool, ClassifierEntity classifierEntity) throws Exception {
		Prediction prediction = new Prediction();
		for(int idx=0;idx<tokens.size();idx++) {
			// prepare the input for NN
			Example ex = classifierEntity.getExampleFeatures(tokens, idx, false, tool);
			int transition = classifierEntity.sparse.giveTheBestChoice(ex);
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
						Example relationExample = getExampleFeatures(tokens, true, former, latter, tool);
						transition = sparse.giveTheBestChoice(relationExample)+4; // add the offset to simulate a joint decoder
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
				Example relationExample = getExampleFeatures(tokens, true, former, latter, tool);
				int transition = sparse.giveTheBestChoice(relationExample)+4; // add the offset to simulate a joint decoder
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
	
	public ClassifierRelation(Parameters parameters) {
		
		this.parameters = parameters;
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

	@Override
	public double[][] getE() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getPaddingID() {
		// TODO Auto-generated method stub
		return -1;
	}

	@Override
	public int getPositionID(int position) {
		// TODO Auto-generated method stub
		return -1;
	}

	@Override
	public double[][] getEg2E() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public static boolean checkWrongState(Prediction prediction) {
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

}
