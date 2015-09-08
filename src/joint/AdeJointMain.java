package joint;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.regex.Pattern;

import utils.ADESentence;
import utils.Abstract;
import cn.fox.biomedical.Dictionary;
import cn.fox.biomedical.Sider;
import cn.fox.machine_learning.BrownCluster;
import cn.fox.machine_learning.Perceptron;
import cn.fox.machine_learning.PerceptronFeatureFunction;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import cn.fox.nlp.EnglishPos;
import cn.fox.nlp.Segment;
import cn.fox.nlp.Sentence;
import cn.fox.nlp.SentenceSplitter;
import cn.fox.nlp.TokenizerWithSegment;
import cn.fox.utils.Evaluater;
import cn.fox.utils.IoUtils;
import cn.fox.utils.ObjectSerializer;
import cn.fox.utils.ObjectShuffle;
import cn.fox.utils.StopWord;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.FileNameComparator;
import drug_side_effect_utils.RelationEntity;
import drug_side_effect_utils.Tool;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import gnu.trove.TIntArrayList;

public class AdeJointMain {
	public static ArrayList<String> alphabetEntityType = new ArrayList<String>(Arrays.asList("Disease","Chemical"));
	public static ArrayList<String> alphabetRelationType = new ArrayList<String>(Arrays.asList("CID"));
	
	public static void main(String[] args) throws Exception{
		FileInputStream fis = new FileInputStream(args[1]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
		String common_english_abbr = properties.getProperty("common_english_abbr");
		String pos_tagger = properties.getProperty("pos_tagger");
		String wordnet_dict = properties.getProperty("wordnet_dict");
		String parser = properties.getProperty("parser");
		String sider_dict = properties.getProperty("sider_dict");
		String instance_dir = properties.getProperty("instance_dir");
		String jochem_dict = properties.getProperty("jochem_dict");
		String ctdchem_dict = properties.getProperty("ctdchem_dict");
		String ctdmedic_dict = properties.getProperty("ctdmedic_dict");
		String chemical_element_abbr = properties.getProperty("chemical_element_abbr");
		String drug_dict = properties.getProperty("drug_dict");
		String disease_dict = properties.getProperty("disease_dict");
		String disease_max_len = properties.getProperty("disease_max_len");
		String chemical_max_len = properties.getProperty("chemical_max_len");
		String perceptron_ser = properties.getProperty("perceptron_ser");
		String abstract_dir = properties.getProperty("abstract_dir");
		String train_instance_dir = properties.getProperty("train_instance_dir");
		String test_instance_dir = properties.getProperty("test_instance_dir");
		String use_system_out = properties.getProperty("use_system_out");
		String converge_threshold = properties.getProperty("converge_threshold");
		String max_weight = properties.getProperty("max_weight");
		String brown_cluster_path = properties.getProperty("brown_cluster_path");
		String stop_word = properties.getProperty("stop_word");
		String group = properties.getProperty("group");
		
		TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "ptb3Escaping=false");
		SentenceSplitter sentSplit = new SentenceSplitter(new Character[]{';'}, false, common_english_abbr);
		MaxentTagger tagger = new MaxentTagger(pos_tagger);
		Morphology morphology = new Morphology();
		LexicalizedParser lp = LexicalizedParser.loadModel(parser);
		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
	    GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
	    IDictionary dict = new edu.mit.jwi.Dictionary(new URL("file", null, wordnet_dict));
		dict.open();
		Sider sider = new Sider(sider_dict);
		Pattern complexNounPattern = Pattern.compile("[a-zA-Z0-9][a-zA-Z0-9',\\(\\)\\[\\]\\{\\}\\.~\\+]*(-[a-zA-Z0-9',\\(\\)\\[\\]\\{\\}\\.~\\+]+)+[a-zA-Z0-9]");
		Dictionary jochem = new Dictionary(jochem_dict, 6);
		Dictionary ctdchem = new Dictionary(ctdchem_dict, 6);
		Dictionary ctdmedic = new Dictionary(ctdmedic_dict, 6);
		Dictionary chemElem = new Dictionary(chemical_element_abbr, 1);
		Dictionary drugbank = new Dictionary(drug_dict, 6);
		Dictionary humando = new Dictionary(disease_dict, 6);
		BrownCluster entityBC = new BrownCluster(brown_cluster_path, 100);
		StopWord stopWord = new StopWord(stop_word);
		
		Tool tool = new Tool();
		tool.tokenizerFactory = tokenizerFactory;
		tool.sentSplit = sentSplit;
		tool.tagger = tagger;
		tool.morphology = morphology;
		tool.lp = lp;
		tool.gsf = gsf;
		tool.dict = dict;
		tool.sider = sider;
		tool.complexNounPattern = complexNounPattern;
		tool.jochem = jochem;
		tool.ctdchem = ctdchem;
		tool.ctdmedic = ctdmedic;
		tool.chemElem = chemElem;
		tool.drugbank = drugbank;
		tool.humando = humando;
		tool.entityBC = entityBC;
		tool.stopWord = stopWord;
		
				
		// preprocess
		boolean preprocess = false;
		if(preprocess) {
			File fInstanceDir = new File(instance_dir);
			IoUtils.clearDirectory(fInstanceDir);
			
			File fAbstractDir = new File(abstract_dir);
			for(File abstractFile:fAbstractDir.listFiles()) {
				ArrayList<PerceptronInputData> inputDatas = new ArrayList<PerceptronInputData>();
				ArrayList<PerceptronOutputData> outputDatas = new ArrayList<PerceptronOutputData>();
				Abstract ab = (Abstract)ObjectSerializer.readObjectFromFile(abstractFile.getAbsolutePath());
				
				List<Sentence> mySentences = prepareNlpInfo(ab, tool);
				buildInputData(inputDatas, mySentences);
				buildGoldOutputData(ab, outputDatas, mySentences);
				
				File documentDir = new File(instance_dir+"/"+ab.id);
				documentDir.mkdir();
				for(int k=0;k<inputDatas.size();k++) {
					ObjectSerializer.writeObjectToFile(inputDatas.get(k), documentDir+"/"+k+".input");
					ObjectSerializer.writeObjectToFile(outputDatas.get(k), documentDir+"/"+k+".output");
				}
			}
		}
		
		int beamSize = Integer.parseInt(args[2]);
		TIntArrayList d = new TIntArrayList();
		d.add(Integer.parseInt(disease_max_len));
		d.add(Integer.parseInt(chemical_max_len));
		 
		int maxTrainTime = Integer.parseInt(args[3]);
		
		nValidate(10, tool, beamSize, d, instance_dir, perceptron_ser, maxTrainTime, train_instance_dir, test_instance_dir
				, Boolean.parseBoolean(use_system_out) , Float.parseFloat(converge_threshold), Double.parseDouble(max_weight), group);
		
		
		/*trainTestOnDevelopSet(tool, beamSize, d, instance_dir, perceptron_ser, maxTrainTime, train_instance_dir, test_instance_dir
				, Boolean.parseBoolean(use_system_out) , Float.parseFloat(converge_threshold), Double.parseDouble(max_weight));*/
		
				
	}
	
	
	
	
		
	public static void trainTestOnDevelopSet(Tool tool, int beamSize, TIntArrayList d, 
			String instance_dir, String ser, int max_train_times, String train_instance_dir, String test_instance_dir
			, boolean useSystemOut, float converge_threshold, double max_weight) throws Exception {
		System.out.println("beam_size="+beamSize+", disease_max_len="+d.get(0)+", chemical_max_len="+d.get(1)+", train_times="+max_train_times
				+", converge_threshold="+converge_threshold+", max_weight="+max_weight);

		ArrayList<File> trainDirs = new ArrayList<>(Arrays.asList(new File(train_instance_dir).listFiles()));
		ArrayList<File> testDirs = new ArrayList<>(Arrays.asList(new File(test_instance_dir).listFiles()));


		// prepare the training data
		ArrayList<PerceptronInputData> trainInputDatas = new ArrayList<PerceptronInputData>();
		ArrayList<PerceptronOutputData> trainOutputDatas = new ArrayList<PerceptronOutputData>();
		for(int j=0;j<trainDirs.size();j++) {
			List<File> files = Arrays.asList(((File)trainDirs.get(j)).listFiles());
			Collections.sort(files, new FileNameComparator());
			for(File file:files) {
				if(file.getName().indexOf("input") != -1) {
					trainInputDatas.add((PerceptronInputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				} else {
					trainOutputDatas.add((PerceptronOutputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				}
			}
		}
		
		// prepare the test data
		ArrayList<PerceptronInputData> testInputDatas = new ArrayList<PerceptronInputData>();
		ArrayList<PerceptronOutputData> testOutputDatas = new ArrayList<PerceptronOutputData>();
		for(int j=0;j<testDirs.size();j++) {
			List<File> files = Arrays.asList(((File)testDirs.get(j)).listFiles());
			Collections.sort(files, new FileNameComparator());
			for(File file:files) {
				if(file.getName().indexOf("input") != -1) {
					testInputDatas.add((PerceptronInputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				} else {
					testOutputDatas.add((PerceptronOutputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				}
			}
		}
		
		// create a perceptron
		Perceptron perceptron = new Perceptron1(alphabetEntityType, alphabetRelationType, d, converge_threshold, max_weight);
		ArrayList<PerceptronFeatureFunction> featureFunctions1 = new ArrayList<PerceptronFeatureFunction>(
				Arrays.asList(new EntityFeatures(perceptron)));
		ArrayList<PerceptronFeatureFunction> featureFunctions2 = new ArrayList<PerceptronFeatureFunction>(
				Arrays.asList(new RelationFeatures(perceptron)));
		perceptron.setFeatureFunction(featureFunctions1, featureFunctions2);
		perceptron.buildFeatureAlphabet(trainInputDatas, trainOutputDatas, tool);
		System.out.println("begin to train, features "+perceptron.featureAlphabet.size());
		
		// train
		perceptron.trainPerceptron(max_train_times, beamSize, trainInputDatas, trainOutputDatas, tool);
		
		// test
		int countPredictEntity = 0;
		int countTrueEntity = 0;
		int countCorrectEntity = 0;
		int countPredictMesh = 0;
		int countTrueMesh = 0;
		int countCorrectMesh = 0;
		for(int j=0;j<testInputDatas.size();j++) { 
			
			// predict
			ArrayList<Entity> preEntities = new ArrayList<Entity>();
			ArrayList<RelationEntity> preRelationEntitys = new ArrayList<RelationEntity>();
			ArrayList<Entity> goldEntities = new ArrayList<>();
			ArrayList<RelationEntity> goldRelationEntitys = new ArrayList<>();
			
			PerceptronInputData inputdata = testInputDatas.get(j);
			PerceptronStatus returnType = perceptron.beamSearch((PerceptronInputData1)inputdata, null, false, beamSize, tool);
			PerceptronOutputData1 output = (PerceptronOutputData1)returnType.z;
			
			for(int k=0;k<output.segments.size();k++) {
				Entity segment = output.segments.get(k);
				if(segment.type.equals("Disease") || segment.type.equals("Chemical"))
					preEntities.add(segment);
			}
			preRelationEntitys.addAll(output.relations);
			
			// get gold data
			PerceptronOutputData1 goldData = (PerceptronOutputData1)testOutputDatas.get(j);
			for(int k=0;k<goldData.segments.size();k++) {
				Entity segment = goldData.segments.get(k);
				if(segment.type.equals("Disease") || segment.type.equals("Chemical"))
					goldEntities.add(segment);
				
			}
			goldRelationEntitys.addAll(goldData.relations);
			
					
			// evaluate entity first, this should perform before "add mesh", because "add mesh" may delete some entities.
			countPredictEntity+= preEntities.size();
			
			countTrueEntity += goldEntities.size();
			

			for(Entity preEntity:preEntities) {
				for(Entity goldEntity:goldEntities) {
					if(preEntity.equals(goldEntity)) {
						countCorrectEntity++;
						break;
					}
				}
			}
			
					
			// relation
			countPredictMesh += preRelationEntitys.size();
			
			countTrueMesh += goldRelationEntitys.size();
			
			for(RelationEntity pre:preRelationEntitys) {
				for(RelationEntity gold:goldRelationEntitys) {
					if(pre.equals(gold)) {
						countCorrectMesh++;
						break;
					}
				}
			}
						
			
		}
		
		
		double precisionEntity = Evaluater.getPrecisionV2(countCorrectEntity, countPredictEntity);
		double recallEntity  = Evaluater.getRecallV2(countCorrectEntity, countTrueEntity);
		double f1Entity = Evaluater.getFMeasure(precisionEntity, recallEntity, 1);
		System.out.println("entity p,r,f1 are "+precisionEntity+" "+recallEntity+" "+f1Entity); 
		
		double precisionMesh = Evaluater.getPrecisionV2(countCorrectMesh, countPredictMesh);
		double recallMesh  = Evaluater.getRecallV2(countCorrectMesh, countTrueMesh);
		double f1Mesh = Evaluater.getFMeasure(precisionMesh, recallMesh, 1);
		System.out.println("relation p,r,f1 are "+precisionMesh+" "+recallMesh+" "+f1Mesh); 

			
		ObjectSerializer.writeObjectToFile(perceptron, ser);
		
	}
	
	public static void nValidate(int nFold, Tool tool, int beamSize, TIntArrayList d, 
			String instance_dir, String ser, int max_train_times, String train_instance_dir, String test_instance_dir
			, boolean useSystemOut, float converge_threshold, double max_weight, String group) throws Exception {
		
		System.out.println("beam_size="+beamSize+", disease_max_len="+d.get(0)+", chemical_max_len="+d.get(1)+", train_times="+max_train_times
				+", converge_threshold="+converge_threshold+", max_weight="+max_weight);
			
		List[] splitted = null;
		if(group != null) {
			splitted = new ArrayList[10];
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(group), "utf-8"));
			String thisLine = null;
			List<File> groupFile = new ArrayList<File>();
			int groupCount=0;
			while ((thisLine = br.readLine()) != null) {
				if(thisLine.isEmpty()) {
					splitted[groupCount] = groupFile;
					groupCount++;
					groupFile = new ArrayList<File>();
				} else
					groupFile.add(new File(instance_dir+"/"+thisLine));
			}
			splitted[groupCount] = groupFile;
			br.close();
		} else {
			double testPercent = 1.0/nFold;
			double[] proportions = new double[nFold];
			for(int i=0;i<nFold;i++) {
				proportions[i] = testPercent;
			}
			
			File fAbstractDir = new File(instance_dir);
			splitted = ObjectShuffle.split(Arrays.asList(fAbstractDir.listFiles()), proportions);
		}
		
			
		
		double sumPrecisionEntity = 0;
		double sumRecallEntity = 0;
		double sumF1Entity = 0;
		double sumPrecisionMesh = 0;
		double sumRecallMesh = 0;
		double sumF1Mesh = 0;
		
		Perceptron best = null;
		double bestf1 = 0;

		
		for(int i=1;i<=nFold;i++) { // for each fold
			List<File> trainDataDir = new ArrayList<File>();
			List<File> testDataDir = new ArrayList<File>();
			if(nFold!=1) {
				for(int j=0;j<splitted.length;j++) {
					if(j==i-1) {
						for(int k=0;k<splitted[j].size();k++)
							testDataDir.add((File)splitted[j].get(k));
					} else {
						for(int k=0;k<splitted[j].size();k++)
							trainDataDir.add((File)splitted[j].get(k));
					}
				}

			}
			else {
				for(int j=0;j<splitted.length;j++) {
					for(int k=0;k<splitted[j].size();k++) {
						trainDataDir.add((File)splitted[j].get(k));
						testDataDir.add((File)splitted[j].get(k));
					}
					
				}
			}	
			
			
			

			// prepare the training data
			ArrayList<PerceptronInputData> trainInputDatas = new ArrayList<PerceptronInputData>();
			ArrayList<PerceptronOutputData> trainOutputDatas = new ArrayList<PerceptronOutputData>();
			for(int j=0;j<trainDataDir.size();j++) {
				List<File> files = Arrays.asList(((File)trainDataDir.get(j)).listFiles());
				Collections.sort(files, new FileNameComparator());
				for(File file:files) {
					if(file.getName().indexOf("input") != -1) {
						trainInputDatas.add((PerceptronInputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
					} else {
						trainOutputDatas.add((PerceptronOutputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
					}
				}
			}
			
			// create a perceptron
			Perceptron perceptron = new Perceptron1(alphabetEntityType, alphabetRelationType, d, converge_threshold, max_weight);
			ArrayList<PerceptronFeatureFunction> featureFunctions1 = new ArrayList<PerceptronFeatureFunction>(
					Arrays.asList(new EntityFeatures(perceptron)));
			ArrayList<PerceptronFeatureFunction> featureFunctions2 = new ArrayList<PerceptronFeatureFunction>(
					Arrays.asList(new RelationFeatures(perceptron)));
			perceptron.setFeatureFunction(featureFunctions1, featureFunctions2);
			perceptron.buildFeatureAlphabet(trainInputDatas, trainOutputDatas, tool);
			System.out.println("begin to train, features "+perceptron.featureAlphabet.size());
			// train
			perceptron.trainPerceptron(max_train_times, beamSize, trainInputDatas, trainOutputDatas, tool);

			// test
			int countPredictEntity = 0;
			int countTrueEntity = 0;
			int countCorrectEntity = 0;
			int countPredictMesh = 0;
			int countTrueMesh = 0;
			int countCorrectMesh = 0;
			for(int j=0;j<testDataDir.size();j++) { // for each test file
				
				// prepare input
				ArrayList<PerceptronInputData> inputDatas = new ArrayList<PerceptronInputData>();
				ArrayList<PerceptronOutputData> goldDatas = new ArrayList<>();
				List<File> files = Arrays.asList(((File)testDataDir.get(j)).listFiles());
				Collections.sort(files, new FileNameComparator());
				for(File file:files) {
					if(file.getName().indexOf("input") != -1) {
						inputDatas.add((PerceptronInputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
					} else {
						goldDatas.add((PerceptronOutputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
					}
				}
				// predict
				ArrayList<Entity> preEntities = new ArrayList<Entity>();
				ArrayList<RelationEntity> preRelationEntitys = new ArrayList<RelationEntity>();
				ArrayList<Entity> goldEntities = new ArrayList<>();
				ArrayList<RelationEntity> goldRelationEntitys = new ArrayList<>();
				for(int m=0;m<inputDatas.size();m++) {
					PerceptronInputData inputdata = inputDatas.get(m);
					PerceptronStatus returnType = perceptron.beamSearch((PerceptronInputData1)inputdata, null, false, beamSize, tool);
					PerceptronOutputData1 output = (PerceptronOutputData1)returnType.z;
					
					for(int k=0;k<output.segments.size();k++) {
						Entity segment = output.segments.get(k);
						if(segment.type.equals("Disease") || segment.type.equals("Chemical"))
							preEntities.add(segment);
						
					}
					preRelationEntitys.addAll(output.relations);
					
					// get gold data
					PerceptronOutputData1 goldData = (PerceptronOutputData1)goldDatas.get(m);
					for(int k=0;k<goldData.segments.size();k++) {
						Entity segment = goldData.segments.get(k);
						if(segment.type.equals("Disease") || segment.type.equals("Chemical"))
							goldEntities.add(segment);
						
					}
					goldRelationEntitys.addAll(goldData.relations);
				}
				
				
				// evaluate entity first, this should perform before "add mesh", because "add mesh" may delete some entities.
				countPredictEntity+= preEntities.size();
				
				
				countTrueEntity += goldEntities.size();

				for(Entity preEntity:preEntities) {
					for(Entity goldEntity:goldEntities) {
						if(preEntity.equals(goldEntity)) {
							countCorrectEntity++;
							break;
						}
					}
				}

			
				// relation
				countPredictMesh += preRelationEntitys.size();
				
				countTrueMesh += goldRelationEntitys.size();
				
				for(RelationEntity pre:preRelationEntitys) {
					for(RelationEntity gold:goldRelationEntitys) {
						if(pre.equals(gold)) {
							countCorrectMesh++;
							break;
						}
					}
				}

			}
			
			
			double precisionEntity = Evaluater.getPrecisionV2(countCorrectEntity, countPredictEntity);
			double recallEntity  = Evaluater.getRecallV2(countCorrectEntity, countTrueEntity);
			double f1Entity = Evaluater.getFMeasure(precisionEntity, recallEntity, 1);
			System.out.println(i+" fold: entity p,r,f1 are "+precisionEntity+" "+recallEntity+" "+f1Entity); 
			
			double precisionMesh = Evaluater.getPrecisionV2(countCorrectMesh, countPredictMesh);
			double recallMesh  = Evaluater.getRecallV2(countCorrectMesh, countTrueMesh);
			double f1Mesh = Evaluater.getFMeasure(precisionMesh, recallMesh, 1);
			System.out.println(i+" fold: Mesh p,r,f1 are "+precisionMesh+" "+recallMesh+" "+f1Mesh); 
			
			
			sumPrecisionEntity += precisionEntity;
			sumRecallEntity += recallEntity;
			sumF1Entity += f1Entity;
			sumPrecisionMesh += precisionMesh;
			sumRecallMesh += recallMesh;
			sumF1Mesh += f1Mesh;
			
			
			if(f1Mesh>bestf1) {
				bestf1 = f1Mesh;
				best = perceptron;
			}
			
		}
		ObjectSerializer.writeObjectToFile(best, ser);
		System.out.println("The macro average of entity p,r,f1 are "+sumPrecisionEntity/nFold+" "+sumRecallEntity/nFold+" "+sumF1Entity/nFold); 
		System.out.println("The macro average of Mesh p,r,f1 are "+sumPrecisionMesh/nFold+" "+sumRecallMesh/nFold+" "+sumF1Mesh/nFold); 

	}
	
	
	public static void buildGoldOutputData(Abstract ab, ArrayList<PerceptronOutputData> outDatas,List<Sentence> mySentences) throws Exception {
		// build output data
		int k = 0;
		for(ADESentence sentence:ab.sentences) {
			Sentence mySentence = mySentences.get(k);
			PerceptronOutputData1 outputdata = new PerceptronOutputData1(true, mySentence.tokens.size());
			Entity entity = new Entity(null, null, 0, null, null);
			Entity oldGold = null;
			// for each token
			for(int i=0;i<mySentence.tokens.size();i++) {
				CoreLabel token = mySentence.tokens.get(i);
				// build the segments of output data begin
				Entity newGold = isInsideAGoldEntityAndReturnIt(token.beginPosition(), token.endPosition(), sentence);
				if(newGold == null) {
					if(entity.text!=null) { // save the old
						outputdata.segments.add(entity);
						entity = new Entity(null, null, 0, null, null);
					}
					// save the current, because the empty type segment has only one length.
					entity.type = Perceptron.EMPTY;
					entity.offset = token.beginPosition();
					entity.text = token.word();
					entity.start = i;
					entity.end = i;
					outputdata.segments.add(entity);
					entity = new Entity(null, null, 0, null, null);
				} else {
					if(oldGold!=newGold) { // it's a new entity
						if(entity.text!=null) { // save the old
							outputdata.segments.add(entity);
							entity = new Entity(null, null, 0, null, null);
						}
						// it's the begin of a new entity, and we set its information but don't save it,
						// because a entity may be more than one length.
						entity.type = newGold.type;
						entity.offset = token.beginPosition();
						entity.text = token.word();
						entity.start = i;
						entity.end = i;
						entity.mesh = newGold.mesh;
						
						oldGold = newGold;
					} else { // it's a old entity with more than one length
						int whitespaceToAdd = token.beginPosition()-(entity.offset+entity.text.length());
						for(int j=0;j<whitespaceToAdd;j++)
							entity.text += " ";
						// append the old entity with the current token
						entity.text += token.word();
						entity.end = i;	
					}
				}
				// build the segments of output data end
				
			}
			if(entity.text!=null) { // save the old
				outputdata.segments.add(entity);
			}
			// build the relations of output data begin
			if(sentence.relaitons!=null) {
				
				for(RelationEntity old:sentence.relaitons) {
					Entity entity1 = null;
					for(int i=0;i<outputdata.segments.size();i++) {
						if(outputdata.segments.get(i).type.equals(Perceptron.EMPTY))
							continue;
						if(old.entity1.equals(outputdata.segments.get(i)))
							entity1 = outputdata.segments.get(i);
					}
					Entity entity2 = null;
					for(int i=0;i<outputdata.segments.size();i++) {
						if(outputdata.segments.get(i).type.equals(Perceptron.EMPTY))
							continue;
						if(old.entity2.equals(outputdata.segments.get(i)))
							entity2 = outputdata.segments.get(i);
					}
					if(entity1.equals(entity2) || entity1==null || entity2==null)
						throw new Exception();
					RelationEntity relationEntity = new RelationEntity(alphabetRelationType.get(0), entity1, entity2);
					outputdata.relations.add(relationEntity);
				}
			}
			// build the relations of output data end

			outDatas.add(outputdata);
			
			k++;
		}
	}
	
	
	
	public static Entity isInsideAGoldEntityAndReturnIt(int start, int end, ADESentence sentence) {
		for(Entity temp:sentence.entities) {
			if(start >= temp.offset && end <= (temp.offset+temp.text.length()))
			{
				return temp;
			}
		}
		return null;
	}
	
	public static List<Sentence> prepareNlpInfo(Abstract ab, Tool tool) {
		List<Sentence> mySentences = new ArrayList<Sentence>();
		
		for(ADESentence sentence:ab.sentences) {
			ArrayList<Segment> given = new ArrayList<Segment>();
			ArrayList<Segment> segments = TokenizerWithSegment.tokenize(sentence.offset, sentence.text, given);
			List<CoreLabel> tokens = new ArrayList<CoreLabel>();
			for(Segment segment:segments) {
				CoreLabel token = new CoreLabel();
				token.setWord(segment.word);
				token.setValue(segment.word);
				token.setBeginPosition(segment.begin);
				token.setEndPosition(segment.end);
				tokens.add(token);
			}
			
			// pos tagging
			tool.tagger.tagCoreLabels(tokens);
			// lemma
			for(int i=0;i<tokens.size();i++)
				tool.morphology.stem(tokens.get(i));
			// parsing
			Tree root = tool.lp.apply(tokens);
			root.indexLeaves();
			root.setSpans();
			List<Tree> leaves = root.getLeaves();
			// depend
			GrammaticalStructure gs = tool.gsf.newGrammaticalStructure(root);
			List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
		    SemanticGraph depGraph = new SemanticGraph(tdl);
		    
		    Sentence mySentence = new Sentence();
			mySentence.tokens = tokens;
			mySentence.root = root;
			mySentence.leaves = leaves;
			mySentence.depGraph = depGraph;
			mySentences.add(mySentence);
			
		}
		
		return mySentences;
	}
	
	public static CoreLabel getHeadWordOfSegment(Entity segment, PerceptronInputData1 input) {
		if(segment.start==segment.end)
			return input.sentInfo.tokens.get(segment.start);
		else {
			int i=segment.start;
			for(;i<=segment.end;i++){ // in order to find a prep
				if(EnglishPos.getType(input.sentInfo.tokens.get(i).tag()) == EnglishPos.Type.PREP)
					break;
			}
			if(i>segment.end || i==segment.start) { // not find a prep or the first token is prep
				return input.sentInfo.tokens.get(segment.end);
			} else { // use the word before prep as head
				return input.sentInfo.tokens.get(i-1);
			}
			
		}
	}
	
	public static void buildInputData(ArrayList<PerceptronInputData> inputDatas,List<Sentence> mySentences) {
		// build input data
		for(Sentence mySentence:mySentences) {
			PerceptronInputData1 inputdata = new PerceptronInputData1();
			for(int i=0;i<mySentence.tokens.size();i++) {
				CoreLabel token = mySentence.tokens.get(i);
				// build input data
				inputdata.tokens.add(token.word());
				inputdata.offset.add(token.beginPosition());
			}
			inputdata.sentInfo = mySentence;
			inputDatas.add(inputdata);
		}
	}
	
	/*public static void test() {
		File dir = new File("E:/ade/joint_instance");
		List<File> files = Arrays.asList(dir.listFiles());
		
		int group = (int)(files.size()/10.0);
		for(int i=0;i<10;i++) {
			if(i!=9) {
				for(int j=0;j<group;j++) {
					
					String s = files.get(i*group+j).getName();
					System.out.println(s);
					
				}
				
			} else {
				for(int j=0;i*group+j<files.size();j++) {
					
					String s = files.get(i*group+j).getName();
					System.out.println(s);
					
				}
				
			}
			System.out.println();
		}
	}*/
}
