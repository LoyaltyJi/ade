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
import gnu.trove.TIntArrayList;
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
			
			ClassifierRelation classifierRelation = new ClassifierRelation(parameters);
			
			System.out.println(Parameters.SEPARATOR+" group "+i);
			BestPerformance best = classifierRelation.trainAndTest(trainAb, devAb, testAb,modelFile+i, tool, debug, entityModelFile+i);
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
		//example.featureIdx.add(getFeatureId("ENT#"+former.type+"#"+latter.type));
		
		// entity wordnet
		example.featureIdx.add(getFeatureId("SYN#"+ClassifierEntity.getSynset(former.text, tool)+"#"+ClassifierEntity.getSynset(latter.text, tool)));
		example.featureIdx.add(getFeatureId("HYP#"+ClassifierEntity.getHyper(former.text, tool)+"#"+ClassifierEntity.getHyper(latter.text, tool)));
		
		// entity
		example.featureIdx.add(getFeatureId("EN#"+former.text.toLowerCase()+"#"+latter.text.toLowerCase()));
		
				
		return example;
	}
	
	public List<Example> generateTrainExamples(List<Abstract> trainAbs, Tool tool)
			throws Exception  {
		
		List<Example> ret = new ArrayList<>();
				
		for(Abstract ab:trainAbs) { 
			for(ADESentence sentence:ab.sentences) {
				// for each sentence
				List<CoreLabel> tokens = ClassifierEntity.prepareNLPInfo(tool, sentence);
				// resort the entities in the sentence
				List<Entity> entities = Util.resortEntity(sentence);
				// fill 'start' and 'end' of the entities
				//Util.fillEntity(entities, tokens);
				
				Prediction prediction = new Prediction();
				// for each token, we generate an entity example
				for(int idx=0;idx<tokens.size();idx++) {
					// prepare the input for NN
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
					} else if(curTran==3 && ClassifierRelation.checkWrongState(prediction)) { // append the current entity
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
						if((lastTran==3 && ClassifierRelation.checkWrongState(prediction)) || lastTran==1 || lastTran==2) {
							// if curTran 1 or 2, the last entities should not be considered
							int latterIdx = (curTran==1 || curTran==2) ? prediction.entities.size()-2:prediction.entities.size()-1;
							Entity latter = prediction.entities.get(latterIdx);
							for(int j=0;j<latterIdx;j++) {
															
								Entity former = prediction.entities.get(j);
								if(latter.type.equals(former.type))
									continue;
								
								Example relationExample = getExampleFeatures(tokens, true, former, latter, tool);
								double[] relationGoldLabel = {0,0};	
								transition = -1;
								RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
								if(sentence.relaitons.contains(tempRelation)) {
									// connect
									relationGoldLabel[1] = 1;
									transition = 5;
								} else {
									// not connect
									relationGoldLabel[0] = 1;
									transition = 4;
								}
								relationExample.label = relationGoldLabel;
								ret.add(relationExample);
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
				if((curTran==3 && ClassifierRelation.checkWrongState(prediction)) || curTran==1 || curTran==2) {
					int latterIdx = prediction.entities.size()-1;
					Entity latter = prediction.entities.get(latterIdx);

					for(int j=0;j<latterIdx;j++) {
						Entity former = prediction.entities.get(j);
						if(latter.type.equals(former.type))
							continue;
						Example relationExample = getExampleFeatures(tokens, true, former, latter, tool);
						double[] relationGoldLabel = {0,0};	
						int transition = -1;
						RelationEntity tempRelation = new RelationEntity(Parameters.RELATION, former, latter);
						if(sentence.relaitons.contains(tempRelation)) {
							// connect
							relationGoldLabel[1] = 1;
							transition = 5;
						} else {
							// not connect
							relationGoldLabel[0] = 1;
							transition = 4;
						}
						relationExample.label = relationGoldLabel;
						ret.add(relationExample);
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
	
	
	public BestPerformance trainAndTest(List<Abstract> trainAbs, List<Abstract> devAbs, 
			List<Abstract> testAbs, String modelFile, 
			Tool tool, boolean debug, String entityModelFile) 
		throws Exception {
		ClassifierEntity classifierEntity = (ClassifierEntity)ObjectSerializer.readObjectFromFile(entityModelFile);
	    
		// generate training examples
		// generate alphabet simultaneously
		featureIDs = new TObjectIntHashMap<>();
		List<Example> examples = generateTrainExamples(trainAbs, tool);
		freezeAlphabet = true;
		System.out.println("Total sparse feature number: "+featureIDs.size());
		// new a NN and initialize its weight
		sparse  = new SparseLayer(parameters, this, featureIDs.size(), 2, debug, false);
		
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
			
			for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * parameters.batchSize;
				int end_pos = (updateIter + 1) * parameters.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.add(examples.get(indexes.get(idy)));
				}
				
				GradientKeeper keeper = sparse.process(subExamples);
				sparse.updateWeights(keeper);
			}
						
			
			
			if (iter>0 && iter % parameters.evalPerIter == 0) {
				evaluate(tool, devAbs, testAbs, modelFile, best, classifierEntity, false);
			}			
		}
		
		evaluate(tool, devAbs, testAbs, modelFile, best, classifierEntity, true);
		
		return best;
	}
	
	public void evaluate(Tool tool, List<Abstract> devAbs, List<Abstract> testAbs, String modelFile, BestPerformance best
			, ClassifierEntity classifierEntity, boolean printerror )
			throws Exception {
		

		DecodeStatistic stat = new DecodeStatistic();
        for(Abstract devAb:devAbs) {
        	if(printerror) {
        		System.out.println(Parameters.SEPARATOR);
        		System.out.println("Document: "+devAb.id);
        	}
        	for(ADESentence gold:devAb.sentences) {
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
          		List<CoreLabel> tokens = ClassifierEntity.prepareNLPInfo(tool, gold);
          		ADESentence predicted = null;
          		predicted = decode(tokens, tool, classifierEntity);
          		
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
						if(latter.type.equals(former.type))
							continue;
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
				if(latter.type.equals(former.type))
					continue;
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
