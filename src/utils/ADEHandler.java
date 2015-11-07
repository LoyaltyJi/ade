package utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.Sequence;
import cn.fox.biomedical.Dictionary;
import cn.fox.mallet.FeatureVectorMaker;
import cn.fox.mallet.MalletSequenceTaggerInstance;
import cn.fox.nlp.Segment;
import cn.fox.nlp.Sentence;
import cn.fox.nlp.SentenceSplitter;
import cn.fox.nlp.TokenizerWithSegment;
import cn.fox.utils.CharCode;
import cn.fox.utils.ObjectSerializer;
import cn.fox.utils.StopWord;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.LexicalPattern;
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
import gnu.trove.TObjectDoubleHashMap;

public class ADEHandler {
	
	public static enum Label {
		B_D,I_D,B_C,I_C,O
	}

	public static void main(String[] args) throws Exception{
		
		FileInputStream fis = new FileInputStream("E:/biocreative2015/entity.properties");
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
		String common_english_abbr = properties.getProperty("common_english_abbr");
		String pos_tagger = properties.getProperty("pos_tagger");
		String drug_dict = properties.getProperty("drug_dict");
		String disease_dict = properties.getProperty("disease_dict");
		String wordnet_dict = properties.getProperty("wordnet_dict");
		String parser = properties.getProperty("parser");
		String jochem_dict = properties.getProperty("jochem_dict");
		String ctdchem_dict = properties.getProperty("ctdchem_dict");
		String ctdmedic_dict = properties.getProperty("ctdmedic_dict");
		String chemical_element_abbr = properties.getProperty("chemical_element_abbr");
		String stop_word = properties.getProperty("stop_word");
		String entity_recognizer_ser = properties.getProperty("entity_recognizer_ser");
		
		
		LexicalizedParser lp = LexicalizedParser.loadModel(parser);
		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
	    GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		SentenceSplitter sentSplit = new SentenceSplitter(new Character[]{';'},false, common_english_abbr);
		TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "ptb3Escaping=false");
		MaxentTagger tagger = new MaxentTagger(pos_tagger);
		Dictionary drugbank = new Dictionary(drug_dict, 1);
		Dictionary humando = new Dictionary(disease_dict, 1);
		IDictionary dict = new edu.mit.jwi.Dictionary(new URL("file", null, wordnet_dict));
		dict.open();
		Morphology morphology = new Morphology();
		Dictionary jochem = new Dictionary(jochem_dict, 1);
		Dictionary ctdchem = new Dictionary(ctdchem_dict, 1);
		Dictionary ctdmedic = new Dictionary(ctdmedic_dict, 1);
		Dictionary chemElem = new Dictionary(chemical_element_abbr, 1);
		Pattern complexNounPattern = Pattern.compile("[a-zA-Z0-9][a-zA-Z0-9',\\(\\)\\[\\]\\{\\}\\.~\\+]*(-[a-zA-Z0-9',\\(\\)\\[\\]\\{\\}\\.~\\+]+)+[a-zA-Z0-9]");
		StopWord stopWord = new StopWord(stop_word);
		
		Tool tool = new Tool();
		tool.sentSplit = sentSplit;
		tool.tokenizerFactory = tokenizerFactory;
		tool.tagger = tagger;
		tool.drugbank = drugbank;
		tool.humando = humando;
		tool.dict = dict;
		tool.morphology = morphology;
		tool.lp = lp;
		tool.gsf = gsf;
		tool.jochem = jochem;
		tool.ctdchem = ctdchem;
		tool.ctdmedic = ctdmedic;
		tool.chemElem = chemElem;
		tool.complexNounPattern = complexNounPattern;
		tool.stopWord = stopWord;
		
		
		
		
		// build the data source of joint experiment
		/*findOverlappedAndRemove("F:/biomedical resource/ADE-Corpus-V2/DRUG-AE.rel",	"F:/biomedical resource/ADE-Corpus-V2/DRUG-AE-overlap.rel");
		makeAbstract(tool, entity_recognizer_ser, "F:/biomedical resource/ADE-Corpus-V2/DRUG-AE-overlap.rel", null, "E:/ade/abstract");
		
			
		// build the data source of pipeline
		buildNegtiveWithCRF(tool, entity_recognizer_ser, "F:/biomedical resource/ADE-Corpus-V2/ADE-NEG-1644.txt", 
				"F:/biomedical resource/ADE-Corpus-V2/ADE-NEG-1644-crf.txt", "F:/biomedical resource/ADE-Corpus-V2/ADE-NEG-1644-cantfind.txt");
		buildNegtiveWithDict("F:/biomedical resource/ADE-Corpus-V2/ADE-NEG-1644-cantfind.txt",
				"F:/biomedical resource/ADE-Corpus-V2/ADE-NEG-1644-dict.txt",
				"F:/biomedical resource/ADE-Corpus-V2/ADE-NEG-1644-dict-cantfind.txt");*/
		
		makeWordResource("F:/biomedical resource/ADE-Corpus-V2/DRUG-AE.rel", "F:/biomedical resource/ADE-Corpus-V2/ADE-NEG.txt", 
				"F:/biomedical resource/ADE-Corpus-V2/wordresource.txt");
		
		
	}
	
	public static void makeAbstract(Tool tool, String ser, String posFile, String negFile, String abstractDir)  throws Exception{
		CRF crf = (CRF)ObjectSerializer.readObjectFromFile(ser);
		HashMap<String, Abstract> abstracts = new HashMap<>();
		
		BufferedReader positive = new BufferedReader(new InputStreamReader(new FileInputStream(posFile), "utf-8"));
		String thisLine = null;
		while ((thisLine = positive.readLine()) != null && !thisLine.isEmpty()) {
			if(thisLine.indexOf("There have been many reports of probable lithium-induced")!=-1)
				System.out.print("");
			ADELine line = parsingLine(thisLine);
			ADESentence sentence = new ADESentence(line.sentence);
			if(abstracts.keySet().contains(line.id)) {
				// old abstract
				Abstract ab = abstracts.get(line.id);
				if(ab.sentences.contains(sentence)) { // old sentence
					ADESentence old = null;
					for(ADESentence temp:ab.sentences) {
						if(temp.equals(sentence)) {
							old =temp;
							break;
						}
					}
					int aeBegin = line.ae.offset;
					int aeEnd = line.ae.offsetEnd-1; // last index
					int drugBegin = line.drug.offset;
					int drugEnd = line.drug.offsetEnd-1; // last index
					boolean overlap = false; // get rid of the overlap entities in the same sentence, about 15
					for(Entity oldEntity:old.entities) {
						int oldEntityBegin = oldEntity.offset;
						int oldEntityEnd = oldEntity.offsetEnd-1;
						if((!line.ae.text.equals(oldEntity.text) && aeEnd>=oldEntityBegin && aeBegin<=oldEntityEnd)) {
							// overlap
							System.out.println(line.ae+" overlap with "+oldEntity);
							overlap = true;
							break;
						} else if((!line.drug.text.equals(oldEntity.text) && drugEnd>=oldEntityBegin && drugBegin<=oldEntityEnd)) {
							// overlap
							System.out.println(line.drug+" overlap with "+oldEntity);
							overlap = true;
							break;
						} 
					}
					if(overlap==false) {
						old.entities.add(line.ae);
						old.entities.add(line.drug);
						if(old.relaitons.contains(new RelationEntity("CID", line.ae, line.drug)))
							System.out.print("");
						old.relaitons.add(new RelationEntity("CID", line.ae, line.drug));
					}
					
				} else { // new sentence
					sentence.entities = new HashSet<>();
					sentence.entities.add(line.ae);
					sentence.entities.add(line.drug);
					sentence.relaitons = new HashSet<>();
					sentence.relaitons.add(new RelationEntity("CID", line.ae, line.drug));
					// compute the sentence offset
					sentence.offset = -1;
					int candidateAEPosition = sentence.text.indexOf(line.ae.text);
OUT:				while(candidateAEPosition!=-1 ) {
						int candidateDrugPosition = sentence.text.indexOf(line.drug.text);
						while(candidateDrugPosition!=-1) {
							if(Math.abs(candidateAEPosition-candidateDrugPosition) == Math.abs(line.ae.offset-line.drug.offset)) {
								// assume find the real position
								sentence.offset = line.ae.offset-candidateAEPosition;
								break OUT;
							}
							candidateDrugPosition = sentence.text.indexOf(line.drug.text, candidateDrugPosition+line.drug.text.length());
						}
						
						candidateAEPosition = sentence.text.indexOf(line.ae.text, candidateAEPosition+line.ae.text.length());
					}
					if(sentence.offset<0) {
						System.out.println(sentence.text);
						throw new Exception();
					}
					
					ab.sentences.add(sentence);
				}
			} else {
				// new abstract
				Abstract ab = new Abstract();
				ab.id = line.id;
				sentence.entities = new HashSet<>();
				sentence.entities.add(line.ae);
				sentence.entities.add(line.drug);
				sentence.relaitons = new HashSet<>();
				sentence.relaitons.add(new RelationEntity("CID", line.ae, line.drug));
				// compute the sentence offset
				sentence.offset = -1;
				int candidateAEPosition = sentence.text.indexOf(line.ae.text);
OUT:			while(candidateAEPosition!=-1 ) {
					int candidateDrugPosition = sentence.text.indexOf(line.drug.text);
					while(candidateDrugPosition!=-1) {
						if(Math.abs(candidateAEPosition-candidateDrugPosition) == Math.abs(line.ae.offset-line.drug.offset)) {
							// assume find the real position
							sentence.offset = line.ae.offset-candidateAEPosition;
							break OUT;
						}
						candidateDrugPosition = sentence.text.indexOf(line.drug.text, candidateDrugPosition+line.drug.text.length());
					}
					
					candidateAEPosition = sentence.text.indexOf(line.ae.text, candidateAEPosition+line.ae.text.length());
				}
				if(sentence.offset<0) {
					System.out.println(sentence.text);
					throw new Exception();
				}
				ab.sentences.add(sentence);
				abstracts.put(line.id, ab);
			}
		}
		positive.close();
		
		if(negFile != null) {
			BufferedReader negative = new BufferedReader(new InputStreamReader(new FileInputStream(negFile), "utf-8"));

			while ((thisLine = negative.readLine()) != null && !thisLine.isEmpty()) {
				ADELine line = parsingNeg(thisLine);
				// get the entities
				Instance instance = sentenceToInstance(crf.getInputAlphabet(), line.sentence, tool);
				Sequence preOutput = crf.transduce((Sequence)instance.getData());
				ArrayList<Entity> preEntities = new ArrayList<>();
				decode(preOutput, preEntities, (List<CoreLabel>)instance.getSource(), tool, line.id, true);
				postprocessGlobal(tool, preEntities);
				
				ADESentence sentence = new ADESentence(line.sentence);
				if(abstracts.keySet().contains(line.id)) {
					// old abstract
					Abstract ab = abstracts.get(line.id);
					if(ab.sentences.contains(sentence)) { // old sentence
						ADESentence old = null;
						for(ADESentence temp:ab.sentences) {
							if(temp.equals(sentence)) {
								old =temp;
								break;
							}
						}
						for(Entity temp:preEntities) {
							old.entities.add(temp);
						}
					} else { // new sentence
						sentence.entities = new HashSet<>();
						for(Entity temp:preEntities) {
							sentence.entities.add(temp);
						}
						ab.sentences.add(sentence);
					}
				} else {
					// new abstract
					Abstract ab = new Abstract();
					ab.id = line.id;
					sentence.entities = new HashSet<>();
					for(Entity temp:preEntities) {
						sentence.entities.add(temp);
					}
					ab.sentences.add(sentence);
					abstracts.put(line.id, ab);
				}
			}
			negative.close();
		}
		
		// dump out all the abstracts
		for(String id:abstracts.keySet()) {
			Abstract ab = abstracts.get(id);
			ObjectSerializer.writeObjectToFile(ab, abstractDir+"/"+id+".abstract");
		}
		
		
	}
	
	// get rid of the overlap between disease and chemical, about totally 120
	public static void findOverlappedAndRemove(String input, String output) throws Exception {
		OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(output), "utf-8");
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(input), "utf-8"));
		String thisLine = null;
		int ctline = 0;
		// test
		while ((thisLine = br.readLine()) != null) {
			if(thisLine.isEmpty()) continue;
			
			ADELine line = parsingLine(thisLine);
			
			int aeBegin = line.ae.offset;
			int aeEnd = line.ae.offsetEnd-1; // last index
			int drugBegin = line.drug.offset;
			int drugEnd = line.drug.offsetEnd-1; // last index
			if((aeEnd>=drugBegin && aeBegin<=drugEnd) || (aeEnd==drugEnd && aeBegin==drugBegin)) {
				// System.out.println(thisLine);
				ctline++;
			} else {
				osw.write(thisLine+"\n");
			}
		}
		System.out.println(ctline);
		br.close();
		osw.close();
	}
	
		
	// find the pair of ae and drug to make a ADELine
	public static void buildNegtiveWithCRF(Tool tool, String ser, String input, String output, String cannotFind) throws Exception {

		CRF crf = (CRF)ObjectSerializer.readObjectFromFile(ser);
		
		OutputStreamWriter oswCantfind = new OutputStreamWriter(new FileOutputStream(cannotFind), "utf-8");
		OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(output), "utf-8");
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(input), "utf-8"));
		String thisLine = null;
		int ctline = 0;

		// test
		while ((thisLine = br.readLine()) != null && !thisLine.isEmpty() /*&& ctline<maxNumber*/) {
			// 6460590 NEG Clioquinol intoxication occurring in the treatment of acrodermatitis enteropathica with reference to SMON outside of Japan.
			int pos = thisLine.indexOf(" "); // after id
			
			String id = thisLine.substring(0, pos);
			
			pos = thisLine.indexOf(" ", pos+1); // after NEG
			
			String sentence = thisLine.substring(pos+1, thisLine.length());
			
			Instance instance = sentenceToInstance(crf.getInputAlphabet(), sentence, tool);
			Sequence preOutput = crf.transduce((Sequence)instance.getData());
			ArrayList<Entity> preEntities = new ArrayList<>();
			
			decode(preOutput, preEntities, (List<CoreLabel>)instance.getSource(), tool, id, true);

			postprocessGlobal(tool, preEntities);
			
			ArrayList<Entity> aeEntities = new ArrayList<>();
			ArrayList<Entity> drugEntities = new ArrayList<>();
			
			for(Entity temp:preEntities) {
				if(temp.type.equals("Disease") ) {
					aeEntities.add(temp);
				} else if(temp.type.equals("Chemical")) {
					drugEntities.add(temp);
				}
			}
			
			if(aeEntities.size()==0 || drugEntities.size()==0) {
				oswCantfind.write(thisLine+"\n");
				continue;
			}
			
			for(Entity ae:aeEntities)
				for(Entity drug:drugEntities) {
					osw.write(id+"|"+sentence+"|"+ae.text+"|"+ae.offset+"|"+(ae.offset+ae.text.length())+"|"+drug.text
						+"|"+drug.offset+"|"+(drug.offset+drug.text.length())+"\n");
					ctline++;
					System.out.println("line:"+ctline);
					
				}
			
			
		}
		
		br.close();
		osw.close();
		oswCantfind.close();
	
	}
	
	
	
	public static ADELine parsingNeg(String thisLine) {
		int pos = thisLine.indexOf(" "); // after id
		
		String id = thisLine.substring(0, pos);
		
		pos = thisLine.indexOf(" ", pos+1); // after NEG
		
		String sentence = thisLine.substring(pos+1);
		
		return new ADELine(id, sentence, null, null);
	}
	
	public static ADELine parsingLine(String thisLine) {
		// 10030778|Intravenous azithromycin-induced ototoxicity.|ototoxicity|43|54|azithromycin|22|34
		int pos = thisLine.indexOf("|"); // after id
		
		String id = thisLine.substring(0, pos);
		
		int sentenceBeginOffset = pos+1;
		
		pos = thisLine.indexOf("|", pos+1); // after sentence
		
		String sentence = thisLine.substring(sentenceBeginOffset, pos);
		int aeTextOffset = pos+1;
				
		pos = thisLine.indexOf("|", pos+1); // after Adverse-Effect
		
		String ae = thisLine.substring(aeTextOffset, pos);
		int aeBeginOffset = pos+1;			
		
		pos = thisLine.indexOf("|", pos+1); // after Begin offset of Adverse-Effect
		
		String strAeBegin = thisLine.substring(aeBeginOffset, pos);
		int aeEndOffset = pos+1;
		
		pos = thisLine.indexOf("|", pos+1); // after end offset of Adverse-Effect
		
		String strAeEnd = thisLine.substring(aeEndOffset, pos);
		int drugTextOffset = pos+1;
		
		pos = thisLine.indexOf("|", pos+1); // after Drug
		
		String drug = thisLine.substring(drugTextOffset, pos);
		int drugBeginOffset = pos+1;
					
		pos = thisLine.indexOf("|", pos+1); // after Begin offset of Drug
		
		String strDrugBegin = thisLine.substring(drugBeginOffset, pos);
		int drugEndOffset = pos+1;
		 // end offset of Drug
		
		String strDrugEnd = thisLine.substring(drugEndOffset, thisLine.length());
		
		int aeBegin = Integer.parseInt(strAeBegin);
		int aeEnd = Integer.parseInt(strAeEnd); 
		int drugBegin = Integer.parseInt(strDrugBegin);
		int drugEnd = Integer.parseInt(strDrugEnd); 
		
		Entity enAe = new Entity(null, "Disease", aeBegin, ae, null);
		enAe.offsetEnd = aeEnd;
		Entity enDrug = new Entity(null, "Chemical", drugBegin, drug, null);
		enDrug.offsetEnd = drugEnd;
		return new ADELine(id, sentence, enAe, enDrug);
	}
	
	
	
	
	public static void buildNegtiveWithDict(String input, String output, String cannotFind) throws Exception {
		int maxTokenNumOfEntity = 5;
		
		Dictionary drugbank = new Dictionary("F:\\biomedical resource\\drugbank.dict", maxTokenNumOfEntity);
		Dictionary humando = new Dictionary("F:\\biomedical resource\\HumanDO.dict", maxTokenNumOfEntity);
		Dictionary jochem = new Dictionary("F:\\biomedical resource\\Jochem\\Jochem.dict", maxTokenNumOfEntity);
		Dictionary ctdchem = new Dictionary("F:\\biomedical resource\\CTD\\CTD_chemicals.dict", maxTokenNumOfEntity);
		Dictionary ctdmedic = new Dictionary("F:\\biomedical resource\\CTD\\CTD_diseases.dict", maxTokenNumOfEntity);
		Dictionary chemElem = new Dictionary("F:\\biomedical resource\\chemical_element_abbr.txt", maxTokenNumOfEntity);
		Tool tool = new Tool();
		tool.drugbank = drugbank;
		tool.humando = humando;
		tool.jochem = jochem;
		tool.ctdchem = ctdchem;
		tool.ctdmedic = ctdmedic;
		tool.chemElem = chemElem;
		StopWord stopWord = new StopWord("E:/biocreative2015/stopword.txt");
		tool.stopWord = stopWord;
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(input), "utf-8"));
		String thisLine = null;
		
		OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(output), "utf-8");
		int count1 = 0;
		int count2 = 0;
		int count3 = 0;
		OutputStreamWriter osw1 = new OutputStreamWriter(new FileOutputStream(cannotFind), "utf-8");
		
		while ((thisLine = br.readLine()) != null && !thisLine.isEmpty()) {
			ADELine line = parsingNeg(thisLine);
			
			ArrayList<Entity> oldEntities = new ArrayList<Entity>();
			// recognize the drug and disease
			ArrayList<Entity> aes = new ArrayList<Entity>();
			ArrayList<Entity> drugs = new ArrayList<Entity>();
			
			
			findEntityInSentenceWithDict(oldEntities, line.sentence, tool, maxTokenNumOfEntity);
			
			for(Entity old:oldEntities) {
				if(old.type.equals("Disease"))
					aes.add(old);
				else {
					drugs.add(old);
				}
			}
			
			if(aes.size()==0 || drugs.size()==0) {
				count1++;
				osw1.write(thisLine+"\n");
			}
			else {
				for(Entity ae:aes) {
					for(Entity drug:drugs) {
						osw.write(line.id+"|"+line.sentence+"|"+ae.text+"|"+ae.offset+"|"+ae.offsetEnd+"|"+drug.text
								+"|"+drug.offset+"|"+drug.offsetEnd+"\n");
					}
				}
				count3++;
			}
				
			
			
		}
		br.close();
		osw.close();
		osw1.close();
		System.out.println(count1+" "+count2+" "+count3);
	}
	
	
	
	public static Instance sentenceToInstance(Alphabet features, String sentence, Tool tool) throws Exception {
		ArrayList<Segment> given = new ArrayList<Segment>();
		Sentence sent = new Sentence();
		ArrayList<Segment> segments = TokenizerWithSegment.tokenize(0, sentence, given);
		List<CoreLabel> tokens = new ArrayList<CoreLabel>();
		for(Segment segment:segments) {
			CoreLabel token = new CoreLabel();
			token.setWord(segment.word);
			token.setValue(segment.word);
			token.setBeginPosition(segment.begin);
			token.setEndPosition(segment.end);
			tokens.add(token);
		}
		
		// for each word
		int sentenceLength = sentence.length();
		// pos tagging
		tool.tagger.tagCoreLabels(tokens);
		// lemma
		for(int i=0;i<tokens.size();i++) {
			tool.morphology.stem(tokens.get(i));
		}
		// parsing
		Tree root = tool.lp.apply(tokens);
		root.indexLeaves();
		root.setSpans();
		List<Tree> leaves = root.getLeaves();
		
		// depend
		GrammaticalStructure gs = tool.gsf.newGrammaticalStructure(root);
		List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
	    SemanticGraph depGraph = new SemanticGraph(tdl);
		
		sent.tokens = tokens;
		sent.root = root;
		sent.leaves = leaves;
		sent.depGraph = depGraph;
		
		FeatureVector[] fvs = new FeatureVector[tokens.size()];
		for(int i=0;i<tokens.size();i++) {
			CoreLabel token = tokens.get(i);
			
			// add features to the instance
			TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<String>();
			prepareFeatures(tool, map, sent, i, null);
			fvs[i] = FeatureVectorMaker.make(features, map);
		
		}
		Instance instance = new Instance(new FeatureVectorSequence(fvs), null, null, tokens);
		return 		instance;
		
	}
	

	public static void findEntityInSentenceWithDict(ArrayList<Entity> oldEntities, String text, Tool tool, int maxTokenNumOfEntity) {
		
		ArrayList<Segment> given = new ArrayList<Segment>();
		ArrayList<Segment> segments = TokenizerWithSegment.tokenize(0, text, given);
		
		for(int begin=0;begin<segments.size();) {
			
			int end=begin;
			for(;end-begin<maxTokenNumOfEntity && end<segments.size();end++) {
				// make a entry
				String entry = "";
				for(int i=begin;i<=end;i++) {
					if(i==begin)
						entry += segments.get(i).word;
					else {
						int whitespaceToAdd = segments.get(i).begin-segments.get(i-1).end;
						for(int q=1;q<=whitespaceToAdd;q++)
							entry += " ";
						entry += segments.get(i).word;
					}
				}
				// search the entry in the dictionaries
				if((tool.humando.contains(entry) || tool.ctdmedic.contains(entry)) 
						&& !tool.stopWord.contains(entry)) {
					boolean overlap = false;
					int offsetBegin = segments.get(begin).begin;
					int offsetEnd = segments.get(end).end;
					for(Entity old:oldEntities) {
						if((offsetEnd>=old.offset && offsetBegin<=old.offsetEnd)) {
							overlap = true;
							break;
						}
					}
					if(!overlap) {
						Entity ae = new Entity(null, "Disease", offsetBegin, entry, null);
						ae.offsetEnd = offsetEnd;
						oldEntities.add(ae);
						break;
					}
				}
				else if((tool.chemElem.containsCaseSensitive(entry) || tool.drugbank.contains(entry) ||
						tool.jochem.contains(entry) || tool.ctdchem.contains(entry))
						&& !tool.stopWord.contains(entry)) {
					boolean overlap = false;
					int offsetBegin = segments.get(begin).begin;
					int offsetEnd = segments.get(end).end;
					for(Entity old:oldEntities) {
						if((offsetEnd>=old.offset && offsetBegin<=old.offsetEnd)) {
							overlap = true;
							break;
						}
					}
					if(!overlap) {
						Entity ae = new Entity(null, "Chemical", offsetBegin, entry, null);
						ae.offsetEnd = offsetEnd;
						oldEntities.add(ae);
						break;
					}
				}
			}
			
			
			if(end-begin==maxTokenNumOfEntity) {
				begin++;
			}
			else {
				begin = end+1;
			}
		}
		
		
	}

	public static void decode(Sequence preOutput, ArrayList<Entity> preEntities, List<CoreLabel> tokens, Tool tool, 
		String documentID, boolean isPostProcess) throws Exception {
		Entity temp = null;
		String lastLabel = "";
		for(int k=0;k<preOutput.size();k++) {
			if(preOutput.get(k).equals(Label.B_D.toString()) || preOutput.get(k).equals(Label.B_C.toString())) {
				if(temp!=null) {
					preEntities.add(temp);
					temp=null;
				}
				String type = preOutput.get(k).equals(Label.B_C.toString()) ? "Chemical":"Disease";
				temp = new Entity(null, type, tokens.get(k).beginPosition(), tokens.get(k).word(),null);
			} else if(preOutput.get(k).equals(Label.I_D.toString())) {
				if(temp == null) {
					// error predication(I_D come first)
					System.out.println("error predication(I_D come first): "+documentID+" "+tokens.get(k).beginPosition()+" "+tokens.get(k).word());
				} else if(lastLabel.equals(Label.B_C.toString()) || lastLabel.equals(Label.I_C.toString())) {
					// error predication(unconsistent)
					System.out.println("error predication(unconsistent I_D): "+documentID+" "+tokens.get(k).beginPosition()+" "+tokens.get(k).word());
					if(temp!=null) { // but we still keep the previous result
						preEntities.add(temp);
						temp=null;
					}
				} else {							
					int whitespaceToAdd = 0;
					if(k>=1) 
						whitespaceToAdd = tokens.get(k).beginPosition()-tokens.get(k-1).endPosition();
					for(int q=1;q<=whitespaceToAdd;q++)
						temp.text += " ";
					temp.text += tokens.get(k).word();
				} 
			} else if(preOutput.get(k).equals(Label.I_C.toString())) {
				if(temp == null) {
					// error predication(I_C come first)
					System.out.println("error predication(I_C come first): "+documentID+" "+tokens.get(k).beginPosition()+" "+tokens.get(k).word());
				} else if(lastLabel.equals(Label.B_D.toString()) || lastLabel.equals(Label.I_D.toString())) {
					// error predication(unconsistent)
					System.out.println("error predication(unconsistent I_C): "+documentID+" "+tokens.get(k).beginPosition()+" "+tokens.get(k).word());
					if(temp!=null) { // but we still keep the previous result
						preEntities.add(temp);
						temp=null;
					}
				} else {							
					int whitespaceToAdd = 0;
					if(k>=1) 
						whitespaceToAdd = tokens.get(k).beginPosition()-tokens.get(k-1).endPosition();
					for(int q=1;q<=whitespaceToAdd;q++)
						temp.text += " ";
					temp.text += tokens.get(k).word();
				} 
				
			} else if(preOutput.get(k).equals(Label.O.toString())){
				if(temp!=null) {
					preEntities.add(temp);
					temp=null;
				}
				// if the label is O, we use some rules to improve the recall
				if(isPostProcess)
					postprocessLocal(tokens, preOutput, k, tool, preEntities);
			} else {
				throw new Exception("wrong state: "+preOutput.get(k));
			}
		
			lastLabel = (String)preOutput.get(k);
		}
		
		if(temp!=null) {
			preEntities.add(temp);
		}
	}
	
	public static void postprocessGlobal(Tool tool, ArrayList<Entity> preEntities) {
		ArrayList<Entity> toBeDeleted = new ArrayList<>();
		
		
		for(Entity pre:preEntities) {
			if(tool.stopWord.contains(pre.text))
				toBeDeleted.add(pre);
			else if(pre.text.length()==1 && CharCode.isLowerCase(pre.text.charAt(0)))
				toBeDeleted.add(pre);
			/*else if(LexicalPattern.getNumNum(pre.text) == pre.text.length()) // all number
				toBeDeleted.add(pre);*/
			else if(LexicalPattern.getAlphaNum(pre.text)==0)
				toBeDeleted.add(pre);
		}
		for(Entity delete:toBeDeleted) {
			preEntities.remove(delete);
		}
	}
	
	// don't need to use StopWord here, we will delete them in the  postprocessGlobal
	public static void postprocessLocal(List<CoreLabel> tokens, Sequence preOutput, int k, Tool tool, ArrayList<Entity> preEntities) {
		CoreLabel token = tokens.get(k);
		/*if(token.word().equals("LA") && token.beginPosition()==221)
			System.out.print("");*/
		// search chem element
		if(tool.chemElem.containsCaseSensitive(token.word())) {
			preEntities.add(new Entity(null, "Chemical", token.beginPosition(), token.word(),null));
			return;
		}
		// search chem dict
		if((tool.drugbank.contains(token.word()) || tool.drugbank.contains(token.lemma()) ||
				tool.jochem.contains(token.word()) || tool.jochem.contains(token.lemma()) ||
				tool.ctdchem.contains(token.word()) || tool.ctdchem.contains(token.lemma())) &&
				!tool.stopWord.contains(token.word())) {
			preEntities.add(new Entity(null, "Chemical", token.beginPosition(), token.word(),null));
			return;
		} 
		// search disease dict
		if((tool.humando.contains(token.word()) || tool.humando.contains(token.lemma()) ||
				tool.ctdmedic.contains(token.word()) || tool.ctdmedic.contains(token.lemma())) &&
				!tool.stopWord.contains(token.word())) {
			preEntities.add(new Entity(null, "Disease", token.beginPosition(), token.word(),null));
			return;
		} 
		// trigger word: if a token contains "-"+trigger, it may be a chemical entity
		String word = token.word().toLowerCase();
		String lemma = token.lemma().toLowerCase();
		String[] triggers = new String[]{"-induced","-associated","-related"};
		for(String trigger:triggers) {
			int pos = -1;
			if((pos = word.indexOf(trigger)) != -1 || (pos = lemma.indexOf(trigger)) != -1) {
				if(token.word().charAt(pos-1) == ')')
					pos --;
				String s = token.word().substring(0,pos);
				if(!tool.stopWord.contains(s)) {
					preEntities.add(new Entity(null, "Chemical", token.beginPosition(), s, null));
					return;
				}
			}
		}
		// coreference: if a token has been regonized as a entity before, it should be now
		for(Entity pre:preEntities) {
			if(pre.text.equalsIgnoreCase(token.word()) || pre.text.equalsIgnoreCase(token.lemma())) {
				preEntities.add(new Entity(null, pre.type, token.beginPosition(), token.word(),null));
				return;
			}
		}
		//  length > 1
		if(token.word().length()>1 && /*it's a all upper token */
				(LexicalPattern.getUpCaseNum(token.word()) == token.word().length() || // has a "(" and ")" around it
				((k>0 && tokens.get(k-1).word().equals("(") && k<tokens.size()-1 && tokens.get(k+1).word().equals(")"))))
				) {
			if(!preEntities.isEmpty()) {
				// the type is the same with the pre-closest entity
				Entity pre = preEntities.get(preEntities.size()-1);
				// if each letter of the token is in the previous entity
				char[] letters = token.word().toLowerCase().toCharArray();
				String preText = pre.text.toLowerCase();
				int i=0;
				int from = 0;  // record the matched postion
				for(;i<letters.length;i++) {
					if((from=preText.indexOf(letters[i], from)) == -1) {
						break;
					} else
						from++;
				}
				if(i==letters.length) {
					preEntities.add(new Entity(null, pre.type, token.beginPosition(), token.word(),null));
					return;
				}
				
			}
			
		}
		
		
	}
		
	public static void prepareFeatures(Tool tool, TObjectDoubleHashMap<String> map, Sentence sent, int i, 
			MalletSequenceTaggerInstance myInstance) throws Exception{
			
		CoreLabel token = sent.tokens.get(i);
		
		map.put("#WD_"+token.lemma().toLowerCase(),1.0);
		map.put("#POS_"+token.tag(),1.0);
		if(i>=1) {
			map.put("#PRETK_"+sent.tokens.get(i-1).lemma().toLowerCase(),1.0);
			map.put("#PREPOS_"+sent.tokens.get(i-1).tag(), 1.0);
		}
		
		if(i<=sent.tokens.size()-2) {
			map.put("#NEXTTK_"+sent.tokens.get(i+1).lemma().toLowerCase(),1.0);
			map.put("#NEXTPOS_"+sent.tokens.get(i+1).tag(),1.0);
		}
		

		{
			String lem = token.lemma().toLowerCase();
			int len = lem.length()>4 ? 4:lem.length();
			map.put("#PREF_"+lem.substring(0, len),1.0);
			map.put("#SUF_"+lem.substring(lem.length()-len, lem.length()),1.0);
		}
		
		String bcHd = tool.brownCluster.getPrefix(token.lemma());
		map.put("E#HEADBC_"+bcHd, 1.0);
		
		

		if((tool.humando.contains(token.word()) || tool.ctdmedic.contains(token.word())) 
				&& !tool.stopWord.contains(token.word())) {
			map.put("#DICTD_", 1.0);
		}
		if((tool.chemElem.containsCaseSensitive(token.word()) || tool.drugbank.contains(token.word()) ||
				tool.jochem.contains(token.word()) || tool.ctdchem.contains(token.word()))
				&& !tool.stopWord.contains(token.word()))
			map.put("#DICTC_", 1.0);
		
		POS[] poses = {POS.NOUN, POS.ADJECTIVE};
		for(POS pos:poses) {
			ISynset synset = WordNetUtil.getMostSynset(tool.dict, token.lemma().toLowerCase(), pos);
			if(synset!= null) {
				map.put("E#HDSYNS"+"_"+synset.getID(),1.0);
			} 

			ISynset hypernym = WordNetUtil.getMostHypernym(tool.dict, token.lemma().toLowerCase(), pos);
			if(hypernym!= null) {
				map.put("E#HDHYPER"+"_"+hypernym.getID(),1.0);
			}
			
		}
		
		LexicalPattern lpattern = new LexicalPattern();
		lpattern.getAll(token.word());
		if(lpattern.ctUpCase == token.word().length())
			map.put("E#UCASE_", 1.0);
		else if(lpattern.ctUpCase == 0)
			map.put("E#LCASE_", 1.0);
		else
			map.put("E#MCASE_", 1.0);
		
		
		if(lpattern.ctAlpha == 0 && lpattern.ctNum == 0)
			map.put("E#NONUMALPHA", 1.0);
		else if(lpattern.ctAlpha != 0 && lpattern.ctNum == 0)
			map.put("E#ONLYALPHA", 1.0);
		else if(lpattern.ctAlpha == 0 && lpattern.ctNum != 0)
			map.put("E#ONLYNUM", 1.0);
		else
			map.put("E#MIX", 1.0);
		
		if(tool.stopWord.contains(token.word())) // match the stop word
			map.put("E#STOP_", 1.0);
		else
			map.put("E#NOSTOP_", 1.0);
			
		
		
		map.put("E#WDLEN_", token.word().length()/10.0);	
		
	}
	// Build a resource used by word2vec
	public static void makeWordResource(String posFile, String negFile, String output)  throws Exception{
		OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(output), "utf-8");
		
		BufferedReader positive = new BufferedReader(new InputStreamReader(new FileInputStream(posFile), "utf-8"));
		String thisLine = null;
		while ((thisLine = positive.readLine()) != null ) {
			if(thisLine.isEmpty())
				continue;
			ADELine line = parsingLine(thisLine);
			ADESentence sentence = new ADESentence(line.sentence);
			
			ArrayList<Segment> given = new ArrayList<Segment>();
			ArrayList<Segment> segments = TokenizerWithSegment.tokenize(0, sentence.text, given);
			
			for(Segment segment:segments) {
				osw.write(segment.word+" ");
			}
		}
		positive.close();
		
		
		BufferedReader negative = new BufferedReader(new InputStreamReader(new FileInputStream(negFile), "utf-8"));

		while ((thisLine = negative.readLine()) != null && !thisLine.isEmpty()) {
			ADELine line = parsingNeg(thisLine);
						
			ADESentence sentence = new ADESentence(line.sentence);
			
			ArrayList<Segment> given = new ArrayList<Segment>();
			ArrayList<Segment> segments = TokenizerWithSegment.tokenize(0, sentence.text, given);
			
			for(Segment segment:segments) {
				osw.write(segment.word+" ");
			}
		}
		negative.close();
		
		
		
		osw.close();
	}
}
