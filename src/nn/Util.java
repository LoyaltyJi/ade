package nn;

import static java.util.stream.Collectors.toSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;
import cn.fox.biomedical.Dictionary;
import cn.fox.machine_learning.BrownCluster;

import cn.fox.machine_learning.Buckshot;
import cn.fox.machine_learning.KMeans;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.math.Matrix;
import cn.fox.math.Normalizer;
import cn.fox.nlp.Segment;
import cn.fox.nlp.Sentence;
import cn.fox.nlp.TokenizerWithSegment;
import cn.fox.nlp.WordVector;
import cn.fox.stanford.Tokenizer;
import cn.fox.utils.ObjectSerializer;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;
import drug_side_effect_utils.Tool;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.PropertiesUtils;
import gnu.trove.TIntArrayList;
import utils.ADELine;
import utils.ADESentence;
import utils.Abstract;

public class Util {
	public static void main(String[] args) throws Exception {
		FileInputStream fis = new FileInputStream(args[0]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
		Parameters parameters = new Parameters(properties);
		
		//prepareWord2VecCorpus(properties, parameters, "E:/ade/word2vec_corpus.txt");
		       
		//prepareWord2VecCorpus(properties, parameters, "E:/ade/word2vec_corpus_lemma.txt");
		//prepareWord2VecCorpus(properties, parameters, "E:/ade/word2vec_corpus_pattern.txt");
		
		File fAbstractDir = new File(PropertiesUtils.getString(properties, "corpusDir", ""));
		File groupFile = new File(PropertiesUtils.getString(properties, "groupFile", ""));
		String modelFile = PropertiesUtils.getString(properties, "modelFile", "");
		String embedFile = PropertiesUtils.getString(properties, "embedFile", "");
		
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
			
			//prepareTSNE(tool, parameters, trainAb);
			//kmeansWord(tool, parameters, trainAb);
			//prepareEntityTSNE(tool, parameters, trainAb);
			prepare1orMoreWordEntity(tool, parameters, trainAb);
			break;
		}
		
		//findClosestWord("neurotoxicity");
	}
	
	static void prepare1orMoreWordEntity(Tool tool, Parameters parameters, List<Abstract> abs) throws Exception {
		NNADE_word nnade = (NNADE_word)ObjectSerializer.readObjectFromFile("E:\\ade\\nnade.ser0");
		
		ArrayList<String> results = new ArrayList<String>();
		int max = 20;
		HashSet<String> triggers = new HashSet<String>(Arrays.asList(new String[]
		{"induced",}));
		nnade.nn.preCompute();
		
		OutputStreamWriter oswVector = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_vector.txt"), "utf-8");
		OutputStreamWriter oswWord = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_word.txt"), "utf-8");
		DecimalFormat df=new DecimalFormat("0.000000"); 
		
OUT:    for(Abstract ab:abs) {
        	for(ADESentence gold:ab.sentences) {
        		List<CoreLabel> tokens = nnade.prepareNLPInfo(tool, gold);
        		        		
RELATION:        for(RelationEntity relation:gold.relaitons) {
        			Entity former = relation.getFormer();
        			for(int i=0;i<tokens.size();i++) {
        				CoreLabel token = tokens.get(i);
        				if(former.offset==token.beginPosition())
        					former.start = i;
        				if(former.offsetEnd==token.endPosition())
        					former.end = i;
        			}
        			Entity latter = relation.getLatter();
        			for(int i=0;i<tokens.size();i++) {
        				CoreLabel token = tokens.get(i);
        				if(latter.offset==token.beginPosition())
        					latter.start = i;
        				if(latter.offsetEnd==token.endPosition())
        					latter.end = i;
        			}
        			
        			if(latter.start-former.end > 2)
        				continue;
        			

        			int triggerPosition = -1;        			
        			for(int i=former.end+1;i<=latter.start-1;i++) {
        				if(triggers.contains(tokens.get(i).word().toLowerCase())) {
        					triggerPosition = i;
        					break;
        				}
        			}
        			
        			if(triggerPosition == -1)
        				continue RELATION;
        			
        			if(former.end-former.start>=2)
						continue RELATION;
        			
        			if(latter.end-latter.start>=2)
						continue RELATION;
        			
        			if(former.start==former.end) {
        				if(results.contains(tokens.get(former.start).word().toLowerCase()))
        					continue RELATION;
        			} else {
        				if(results.contains(former.text.toLowerCase()))
        					continue RELATION;	
        			}
        			
        			if(latter.start==latter.end) {
        				if(results.contains(tokens.get(latter.start).word().toLowerCase()))
        					continue RELATION;
        			} else {
        				if(results.contains(latter.text.toLowerCase()))
        					continue RELATION;	
        			}
        			
        			for(int i=former.start; i<=former.end;i++) {
						String word = tokens.get(i).word().toLowerCase();
						int index = nnade.wordIDs.get(word);
    					if(index == 0 || index == 1) {
    						continue RELATION;
    					}
					}
        			
        			for(int i=latter.start; i<=latter.end;i++) {
						String word = tokens.get(i).word().toLowerCase();
						int index = nnade.wordIDs.get(word);
    					if(index == 0 || index == 1) {
    						continue RELATION;
    					}
					}
        			
        			System.out.println(relation);
        				
        			String trigger = tokens.get(triggerPosition).word().toLowerCase();
        			if(!results.contains(trigger)) {
        				results.add(trigger);
        				int index = nnade.wordIDs.get(trigger);
        				oswWord.write(trigger+"_#"+"\n");
        				for(int j=0;j<nnade.E[0].length;j++) {
							double temp = nnade.E[index][j];
							if(j==nnade.E[0].length-1) {
								oswVector.write(df.format(temp)+"\n");
							} else
								oswVector.write(df.format(temp)+" ");
        						
        				}
        			}
        				
        			{
        				TIntArrayList entityIdx = new TIntArrayList();
        				for(int i=former.start; i<=former.end;i++) {
        					String word = tokens.get(i).word().toLowerCase();
        					int index = nnade.wordIDs.get(word);
            				entityIdx.add(index);
        				}
	        			results.add(former.text.toLowerCase());
	        			oswWord.write(former.text.replace(' ', '_').toLowerCase()+"_"+former.type.substring(0,1)+"\n");
            			double[] embedding = nnade.nn.entityCNN.forward(entityIdx).emb;	
    					for(int j=0;j<embedding.length;j++) {
    						double temp = embedding[j];
    						if(j==embedding.length-1) {
    							oswVector.write(df.format(temp)+"\n");
    						} else
    							oswVector.write(df.format(temp)+" ");
    					}
        			}
        				
        			{
        				results.add(latter.text.toLowerCase());
	        			TIntArrayList entityIdx = new TIntArrayList();
	        			for(int i=latter.start; i<=latter.end;i++) {
	        				String word = tokens.get(i).word().toLowerCase();
	        				int index = nnade.wordIDs.get(word);
	            			entityIdx.add(index);
	        			}
	        			oswWord.write(latter.text.replace(' ', '_').toLowerCase()+"_"+latter.type.substring(0,1)+"\n");
            			double[] embedding = nnade.nn.entityCNN.forward(entityIdx).emb;	
    					for(int j=0;j<embedding.length;j++) {
    						double temp = embedding[j];
    						if(j==embedding.length-1) {
    							oswVector.write(df.format(temp)+"\n");
    						} else
    							oswVector.write(df.format(temp)+" ");
    						
    					}
        					
        			}
        						

    				if(results.size()>max)
    					break OUT;
        			
        			
        		}
        		
        		
        		
        	}
        }
		
		
		oswVector.close();
		oswWord.close();
	}

	
		static void prepareEntityTSNE(Tool tool, Parameters parameters, List<Abstract> abs) throws Exception {
			NNADE_word nnade = (NNADE_word)ObjectSerializer.readObjectFromFile("E:\\ade\\nnade.ser0");
			/*HashSet<String> results = new HashSet<String>(Arrays.asList(new String[]
			{"induce", "associate", "relate", "male", "female", "it", "describe", "report"}));
			*/
			
			ArrayList<String> results = new ArrayList<String>();
			//ArrayList<String> types = new ArrayList<String>();
			int max = 10;
			HashSet<String> triggers = new HashSet<String>(Arrays.asList(new String[]
			{"induce",/*"associate", "relate", "cause", "develop", "produce", "after", "follow", "result"*/}));
			nnade.nn.preCompute();
			
			OutputStreamWriter oswVector = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_entity_vector.txt"), "utf-8");
			OutputStreamWriter oswWord = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_entity_word.txt"), "utf-8");
			DecimalFormat df=new DecimalFormat("0.000000"); 
			
	OUT:    for(Abstract ab:abs) {
	        	for(ADESentence gold:ab.sentences) {
	        		List<CoreLabel> tokens = nnade.prepareNLPInfo(tool, gold);
	        		//ADESentence predicted = null;
	        		//predicted = nnade.decode(tokens, tool);
	        		
	        		for(RelationEntity relation:gold.relaitons) {
	        			Entity former = relation.getFormer();
	        			for(int i=0;i<tokens.size();i++) {
	        				CoreLabel token = tokens.get(i);
	        				if(former.offset==token.beginPosition())
	        					former.start = i;
	        				if(former.offsetEnd==token.endPosition())
	        					former.end = i;
	        			}
	        			Entity latter = relation.getLatter();
	        			for(int i=0;i<tokens.size();i++) {
	        				CoreLabel token = tokens.get(i);
	        				if(latter.offset==token.beginPosition())
	        					latter.start = i;
	        				if(latter.offsetEnd==token.endPosition())
	        					latter.end = i;
	        			}
	        			if(former.start==former.end || latter.start==latter.end)
	        				continue;
	        			
	        			if(latter.start-former.end > 2)
	        				continue;
	        			
	        			/*if(former.end-former.start>1 || latter.end-latter.start>1)
	        				continue;*/
	        			
	        			int triggerPosition = -1;        			
	        			for(int i=former.end+1;i<=latter.start-1;i++) {
	        				if(triggers.contains(tokens.get(i).lemma().toLowerCase())) {
	        					triggerPosition = i;
	        					break;
	        				}
	        			}
	        			
	        			if(triggerPosition !=-1) {
	        				String trigger = tokens.get(triggerPosition).lemma().toLowerCase();
	        				if(!results.contains(trigger)) {
	        					results.add(trigger);
	        					//types.add("#");
	        					int index = nnade.wordIDs.get(trigger);
	        					if(index == 0 || index == 1) {
	        						System.out.println(trigger);
	        					}
	        					oswWord.write(trigger+"_#"+"\n");
	        					
	        					for(int j=0;j<nnade.E[0].length;j++) {
	        						double temp = nnade.E[index][j];
	        						if(j==nnade.E[0].length-1) {
	        							oswVector.write(df.format(temp)+"\n");
	        						} else
	        							oswVector.write(df.format(temp)+" ");
	        						
	        					}
	        				}
	        				
	        				if(former.start==former.end) {
	        					String word = tokens.get(former.start).lemma().toLowerCase();
	        					if(!results.contains(word)) {
	        						results.add(word);

	        						int index = nnade.wordIDs.get(word);
	            					if(index == 0 || index == 1) {
	            						System.out.println(word);
	            					}
	            					oswWord.write(word+"_"+former.type.substring(0,1)+"\n");
	            					
	            					for(int j=0;j<nnade.E[0].length;j++) {
	            						double temp = nnade.E[index][j];
	            						if(j==nnade.E[0].length-1) {
	            							oswVector.write(df.format(temp)+"\n");
	            						} else
	            							oswVector.write(df.format(temp)+" ");
	            						
	            					}
	        					}
	        				} else {
	        					if(!results.contains(former.text.toLowerCase())) {
	        						results.add(former.text.toLowerCase());
	        						
		        					TIntArrayList entityIdx = new TIntArrayList();
		        					for(int i=former.start; i<=former.end;i++) {
		        						String word = tokens.get(i).lemma().toLowerCase();
		        						int index = nnade.wordIDs.get(word);
		            					if(index == 0 || index == 1) {
		            						System.out.println(word);
		            					}
		            					entityIdx.add(index);
		        					}
		        					oswWord.write(former.text.replace(' ', '_').toLowerCase()+"_"+former.type.substring(0,1)+"\n");
	            					
	            					double[] embedding = nnade.nn.entityCNN.forward(entityIdx).emb;
	            					
	            					for(int j=0;j<embedding.length;j++) {
	            						double temp = embedding[j];
	            						if(j==embedding.length-1) {
	            							oswVector.write(df.format(temp)+"\n");
	            						} else
	            							oswVector.write(df.format(temp)+" ");
	            						
	            					}
	        					}
	        				}
	        				
	        				if(latter.start==latter.end) {
	        					String word = tokens.get(latter.start).lemma().toLowerCase();
	        					if(!results.contains(word)) {
	        						results.add(word);

	        						int index = nnade.wordIDs.get(word);
	            					if(index == 0 || index == 1) {
	            						System.out.println(word);
	            					}
	            					oswWord.write(word+"_"+latter.type.substring(0,1)+"\n");
	            					
	            					for(int j=0;j<nnade.E[0].length;j++) {
	            						double temp = nnade.E[index][j];
	            						if(j==nnade.E[0].length-1) {
	            							oswVector.write(df.format(temp)+"\n");
	            						} else
	            							oswVector.write(df.format(temp)+" ");
	            						
	            					}
	        					}
	        				} else {
	        					if(!results.contains(latter.text.toLowerCase())) {
	        						results.add(latter.text.toLowerCase());
		        					TIntArrayList entityIdx = new TIntArrayList();
		        					for(int i=latter.start; i<=latter.end;i++) {
		        						String word = tokens.get(i).lemma().toLowerCase();
		        						int index = nnade.wordIDs.get(word);
		            					if(index == 0 || index == 1) {
		            						System.out.println(word);
		            					}
		            					entityIdx.add(index);
		        					}
		        					
		        					oswWord.write(latter.text.replace(' ', '_').toLowerCase()+"_"+latter.type.substring(0,1)+"\n");
	            					
	            					double[] embedding = nnade.nn.entityCNN.forward(entityIdx).emb;
	            					
	            					for(int j=0;j<embedding.length;j++) {
	            						double temp = embedding[j];
	            						if(j==embedding.length-1) {
	            							oswVector.write(df.format(temp)+"\n");
	            						} else
	            							oswVector.write(df.format(temp)+" ");
	            						
	            					}
	        					}
	        				}
	        						

	        				if(results.size()>max)
	        					break OUT;
	        			}
	        			
	        		}
	        		
	        		
	        		
	        	}
	        }
			
			
			oswVector.close();
			oswWord.close();
			
		}
		
		
		static void prepareTSNE(Tool tool, Parameters parameters, List<Abstract> abs) throws Exception {
			NNADE_word nnade = (NNADE_word)ObjectSerializer.readObjectFromFile("E:\\ade\\nnade.ser0");
			/*HashSet<String> results = new HashSet<String>(Arrays.asList(new String[]
			{"induce", "associate", "relate", "male", "female", "it", "describe", "report"}));
			*/
			
			ArrayList<String> results = new ArrayList<String>();
			ArrayList<String> types = new ArrayList<String>();
			int max = 10;
			HashSet<String> triggers = new HashSet<String>(Arrays.asList(new String[]
			{"induce",/*"associate", "relate", "cause", "develop", "produce", "after", "follow", "result"*/}));
			nnade.nn.preCompute();

	OUT:    for(Abstract ab:abs) {
	        	for(ADESentence gold:ab.sentences) {
	        		List<CoreLabel> tokens = nnade.prepareNLPInfo(tool, gold);
	        		ADESentence predicted = null;
	        		predicted = nnade.decode(tokens, tool);
	        		
	        		for(RelationEntity relation:predicted.relaitons) {
	        			Entity former = relation.getFormer();
	        			if(former.start!=former.end)
	        				continue;
	        			Entity latter = relation.getLatter();
	        			if(latter.start!=latter.end)
	        				continue;
	        			
	        			if(latter.start-former.end > 2)
	        				continue;
	        			
	        			for(int i=former.end+1;i<=latter.start-1;i++) {
	        				if(triggers.contains(tokens.get(i).lemma().toLowerCase())) {
	        					if(!results.contains(tokens.get(former.start).lemma().toLowerCase())) {
	        						results.add(tokens.get(former.start).lemma().toLowerCase());
	        						types.add(former.type.substring(0,1));
	        					}
	        					if(!results.contains(tokens.get(latter.start).lemma().toLowerCase())) {
	        						results.add(tokens.get(latter.start).lemma().toLowerCase());
	        						types.add(latter.type.substring(0,1));
	        					}
	        					if(!results.contains(tokens.get(i).lemma().toLowerCase())) {
	        						results.add(tokens.get(i).lemma().toLowerCase());
	        						types.add("#");
	        					}
	        					
	        					if(results.size()>=max)
	        						break OUT;
	        				}
	        			}
	        		}
	        		
	        		
	        		
	        	}
	        }
			
			
			
			OutputStreamWriter oswVector = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_vector.txt"), "utf-8");
			OutputStreamWriter oswWord = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_word.txt"), "utf-8");
			//OutputStreamWriter oswType = new OutputStreamWriter(new FileOutputStream("D:/tools/T-SNE-Java-master/tsne-demos/test_type.txt"), "utf-8");
			DecimalFormat df=new DecimalFormat("0.000000"); 
			for(int i=0;i<results.size();i++) {
				String word = results.get(i);
				int index = nnade.wordIDs.get(word);
				if(index == 0 || index == 1) {
					System.out.println(word);
					continue;
				}
				oswWord.write(word+"_"+types.get(i)+"\n");
				//oswType.write(types.get(i)+"\n");
				for(int j=0;j<nnade.E[0].length;j++) {
					double temp = nnade.E[index][j];
					if(j==nnade.E[0].length-1) {
						oswVector.write(df.format(temp)+"\n");
					} else
						oswVector.write(df.format(temp)+" ");
					
				}
				
			}
			oswVector.close();
			oswWord.close();
			//oswType.close();
		}
	
	static void findClosestWord(String w) throws Exception {
		NNADE nnade = (NNADE)ObjectSerializer.readObjectFromFile("E:\\ade\\nnade.ser0");
		
		
		ArrayList<String> words = new ArrayList<>();
		ArrayList<Matrix> vectors = new ArrayList<>();
		Matrix wVector = null;
		for(String word:nnade.knownWords) {
			int index = nnade.wordIDs.get(word);
			if(index == 0 || index == 1) {
				System.out.println(word);
				continue;
			}
			if(w.equals(word)) {
				wVector = new Matrix(1, 50, nnade.E[index]);
				continue;
			}
			
			words.add(word);
			vectors.add(new Matrix(1, 50, nnade.E[index]));
		}
		
		// normalize
		/*for(int i=0;i<vectors.size();i++)
			Normalizer.doVectorNormalizing(vectors.get(i));*/
		
		int max = 100;
		ArrayList<Matrix> mat = new ArrayList<>();
		ArrayList<String> word = new ArrayList<>();
		for(int i=0;i<vectors.size();i++) {
			
			double currentDist = Matrix.distanceEuclidean(vectors.get(i), wVector);
			boolean inserted = false;
			for(int j=0;j<mat.size();j++) {
				double oldDist = Matrix.distanceEuclidean(mat.get(j), wVector);
				if(currentDist < oldDist) {
					mat.add(j, vectors.get(i));
					word.add(j, words.get(i));
					inserted = true;
					break;
				}
				
					
			}
			
			if(inserted == false) {
				mat.add(vectors.get(i));
				word.add(words.get(i));
			}
			if(mat.size()>max) {
				mat.remove(mat.size()-1);
				word.remove(word.size()-1);
			}
			
		}
		// dump result
		
			for(String t:word) {
				System.out.print(t+" ");
			}

		
	}
	
	static void kmeansWord(Tool tool, Parameters parameters, List<Abstract> abs) throws Exception {
		NNADE nnade = (NNADE)ObjectSerializer.readObjectFromFile("E:\\ade\\nnade.ser0");
		HashSet<String> results = new HashSet<String>(Arrays.asList(new String[]
		{"induce", "clozapine", "mtx", "heparin", "colchicine", "methotrexate", "thrombocytopenia", "glaucoma",  "myopathy",  "thyrotoxicosis"}));
				
		/*HashSet<String> results = new HashSet<String>();
		HashSet<String> triggers = new HashSet<String>(Arrays.asList(new String[]
		{"induce", "associate", "relate", "cause", "develop", "produce", "after", "follow", "result"}));
		nnade.nn.preCompute();

        for(Abstract ab:abs) {
        	for(ADESentence gold:ab.sentences) {
        		List<CoreLabel> tokens = nnade.prepareNLPInfo(tool, gold);
        		ADESentence predicted = null;
        		predicted = nnade.decode(tokens, tool);
        		
        		for(RelationEntity relation:predicted.relaitons) {
        			Entity former = relation.getFormer();
        			if(former.start!=former.end)
        				continue;
        			Entity latter = relation.getLatter();
        			if(latter.start!=latter.end)
        				continue;
        			
        			for(int i=former.end+1;i<=latter.start-1;i++) {
        				if(triggers.contains(tokens.get(i).lemma().toLowerCase())) {
        					results.add(tokens.get(former.start).lemma().toLowerCase());
        					results.add(tokens.get(latter.start).lemma().toLowerCase());
        					results.add(tokens.get(i).lemma().toLowerCase());
        				}
        			}
        		}
        		
        		
        		
        	}
        }*/
		
		ArrayList<String> words = new ArrayList<>();
		ArrayList<Matrix> vectors = new ArrayList<>();
		for(String word:results) {
			int index = nnade.wordIDs.get(word);
			if(index == 0 || index == 1) {
				System.out.println(word);
				continue;
			}
			
			words.add(word);
			vectors.add(new Matrix(1, 50, nnade.E[index]));
		}
		
		int k = 2;
		// normalize
		for(int i=0;i<vectors.size();i++)
			Normalizer.doVectorNormalizing(vectors.get(i));
		// do Buckshot
		Buckshot bs = new Buckshot(k, vectors);
		ArrayList<Matrix> centroids = bs.doBuckshot();
		
		// do KMeans
		KMeans mk = new KMeans(k, vectors, centroids, 1000);
		mk.getResults();
		
		OutputStreamWriter osw1 = new OutputStreamWriter(new FileOutputStream("E:\\ade\\kmeans.txt"), "utf-8");
		// dump result, i denotes class, j denotes word
		for(int i=0;i<k;i++) {
			osw1.write("##the "+i+" class :\n");
			int line = 0;
			for(int j=0;j<mk.vectors2classes.length;j++) {
				if(i == mk.vectors2classes[j]) { 
					osw1.write(words.get(j)+" ");
					line++;
					if(line==10) {
						osw1.write("\n");
						line = 0;
					}
				}
			}
			osw1.write("\n\n");
		}
		osw1.close();
		
		int max = 10;
		ArrayList<ArrayList<Matrix>> outMat = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<String>> outWord = new ArrayList<ArrayList<String>>();
		for(int i=0;i<k;i++) {
			outMat.add(new ArrayList<Matrix>());
			outWord.add(new ArrayList<String>());
		}
		for(int i=0;i<mk.vectors2classes.length;i++) {
			ArrayList<Matrix> mat = outMat.get(mk.vectors2classes[i]);
			ArrayList<String> word = outWord.get(mk.vectors2classes[i]);
			
			double currentDist = mk.similarity(vectors.get(i), mk.centroids.get(mk.vectors2classes[i]));
			boolean inserted = false;
			for(int j=0;j<mat.size();j++) {
				double oldDist = mk.similarity(mat.get(j), mk.centroids.get(mk.vectors2classes[i]));
				if(currentDist < oldDist) {
					mat.add(j, vectors.get(i));
					word.add(j, words.get(i));
					inserted = true;
					break;
				}
				
					
			}
			
			if(inserted == false) {
				mat.add(vectors.get(i));
				word.add(words.get(i));
			}
			if(mat.size()>max) {
				mat.remove(mat.size()-1);
				word.remove(word.size()-1);
			}
			
		}
		// dump result, i denotes class, j denotes word
		for(int i=0;i<outWord.size();i++) {
			ArrayList<String> word = outWord.get(i);
			System.out.print("##the "+(i+1)+" class :\n");
			for(String w:word) {
				System.out.print(w+" ");
			}
			System.out.println();
		}
	}
	
	public static double exp(double x) {
		if(x>50) x=50;
		else if(x<-50) x=-50;
		return Math.exp(x);
	}
	
	public static double sigmoid(double x) {
		if(x>50) x=50;
		else if(x<-50) x=-50;
		return 1.0/(1+Math.exp(-x));
	}
	
	public static void fillEntity(List<Entity> entities, List<CoreLabel> tokens) {
		for(Entity entity:entities) {
			for(int i=0;i<tokens.size();i++) {
				CoreLabel token = tokens.get(i);
				if(entity.offset==token.beginPosition())
					entity.start = i;
				if(entity.offsetEnd==token.endPosition())
					entity.end = i;
			}
		}
		
	}
	
	public static List<Entity> resortEntity(ADESentence sentence) {
		List<Entity> entities = new ArrayList<>();
		for(Entity temp:sentence.entities) {
			
			// sort the entity by offset
			int i=0;
			for(;i<entities.size();i++) {
				if(entities.get(i).offset>temp.offset)
					break;
			}
			entities.add(i, temp);
			
		}
		return entities;
	}
	
	public static boolean isInsideAEntity(int start, int end, Entity entity) {
		if(start >= entity.offset && end <= (entity.offset+entity.text.length()))
		{
			return true;
		} else {
			return false;
		}
	}
	
	public static int isInsideAGoldEntityAndReturnIt(int start, int end, List<Entity> entities) {
		for(int i=0;i<entities.size();i++) {
			Entity temp = entities.get(i);
			if(start >= temp.offset && end <= (temp.offset+temp.text.length()))
			{
				return i;
			}
		}
		return -1;
	}
	
	public static boolean isFirstWordOfEntity(Entity entity, CoreLabel token) {
		if(token.beginPosition() == entity.offset)
			return true;
		else 
			return false;
	}
	
	public static boolean isLastWordOfEntity(Entity entity, CoreLabel token) {
		if(entity.offset+entity.text.length() == token.endPosition())
			return true;
		else
			return false;
	}
	
	public static void prepareWord2VecCorpus(Properties properties, Parameters parameters, String output) throws Exception {
		File fAbstractDir = new File(PropertiesUtils.getString(properties, "corpusDir", ""));
		File fOutputFile =new File(output);
		Tokenizer tokenizer = new Tokenizer(true, ' ');	
		MaxentTagger tagger = new MaxentTagger(PropertiesUtils.getString(properties, "pos_tagger", ""));
		Morphology morphology = new Morphology();
		
		OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(fOutputFile), "utf-8");
		
		
		
		
		for(File abstractFile:fAbstractDir.listFiles()) {
			Abstract ab = (Abstract)ObjectSerializer.readObjectFromFile(abstractFile.getAbsolutePath());
			
			for(ADESentence sentence:ab.sentences) {
				List<CoreLabel> tokens = tokenizer.tokenize(sentence.offset, sentence.text);
				tagger.tagCoreLabels(tokens);	
				for(int i=0;i<tokens.size();i++)
					morphology.stem(tokens.get(i));
				for(CoreLabel token:tokens) {
					String temp = NNADE.wordPreprocess(token, parameters);
					osw.write(temp+" ");
					
				}
				osw.write("\n");
								
			}
			
		}
		
		osw.close();
		
		System.out.println("Generate a word2vec corpus.");
		System.out.printf("wordPreprocess = %d%n", parameters.wordPreprocess);
		System.out.printf("embeddingSize = %d%n", parameters.embeddingSize);
	}
	
	public static List<String> generateDict(List<String> str, int cutOff)
	  {
	    Counter<String> freq = new IntCounter<>();
	    for (String aStr : str)
	      freq.incrementCount(aStr);

	    List<String> keys = Counters.toSortedList(freq, false);
	    List<String> dict = new ArrayList<>();
	    for (String word : keys) {
	      if (freq.getCount(word) >= cutOff)
	        dict.add(word);
	    }
	    return dict;
	  }
	
	  public static <T> List<T> getRandomSubList(List<T> input, int subsetSize)
	  {
	    int inputSize = input.size();
	    if (subsetSize > inputSize)
	      subsetSize = inputSize;

	    Random random = new Random(System.currentTimeMillis());
	  
	    for (int i = 0; i < subsetSize; i++)
	    {
	      int indexToSwap = i + random.nextInt(inputSize - i);
	      T temp = input.get(i);
	      input.set(i, input.get(indexToSwap));
	      input.set(indexToSwap, temp);
	    }
	    return input.subList(0, subsetSize);
	  }
	
	
}

class Evaluate {
	double loss;
	double correct;
}
