package sparse_pipeline;

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
		}
		
		//findClosestWord("neurotoxicity");
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
	
	  public static final int COOR_WINDOW = 9;
	  public static final HashSet<String> coor = new HashSet<>(Arrays.asList("and","or","whereas","with",
			  "accompanied", "which"));
	  
	  public static Entity getClosestCoorEntity(List<CoreLabel> tokens, int idx, Prediction predict) {
		  int isCoor = 0;
		  int i = idx-1;
		  for(;i>=0;i--) {
			  if(coor.contains(tokens.get(i).word().toLowerCase())) {
				  isCoor = 1;
				  break;
			  }
			 
		  } 
		  
		  if(isCoor!=0) {
			  int begin = idx-COOR_WINDOW;
			  int end = i-1;
			  if(end>=begin) {
				  for(int j=predict.entities.size()-1;j>=0;j--) {
					  if(predict.entities.get(j).end>=begin && predict.entities.get(j).end<=end)
						  return predict.entities.get(j);
				  }
				  return null;
			  } else
				  return null;
 
		  } else
			  return null;
	  }
	  
	  public static Entity getClosestCoorEntity(List<CoreLabel> tokens, 
			  Entity current, Prediction predict) {
		  int isCoor = 0;
		  int i=current.start-1;
		  for(;i>=0;i--) {
			  if(coor.contains(tokens.get(i).word().toLowerCase())) {
				  isCoor = 1;
				  break;
			  }
		  } 
		  
		  
		  if(isCoor!=0) {
			  
			  int begin = current.start-COOR_WINDOW;
			  int end = i-1;
			  if(end>=begin) {
				  for(int j=predict.entities.size()-1;j>=0;j--) {
					  if(predict.entities.get(j).equals(current))
						  continue;
					  if(predict.entities.get(j).end>=begin && predict.entities.get(j).end<=end)
						  return predict.entities.get(j);
				  }
				  return null;
			  } else
				  return null;
		  } else
			  return null;
		  
	  }
  
	  public static Entity getPreviousEntity(List<CoreLabel> tokens, int idx, Prediction predict) {
		  if(predict.entities.isEmpty())
			  return null;
		  else {
			  for(int j=predict.entities.size()-1;j>=0;j--) {
				  if(predict.entities.get(j).end < idx)
					  return predict.entities.get(j);
			  }
			  return null;
		  }

 
	  }
	  
	  // in 2 tokens(include 2)
	  public static Entity getNeighborEntity(List<CoreLabel> tokens, int idx, Prediction predict) {
		  if(predict.entities.isEmpty())
			  return null;
		  else {
			  int begin = idx-2;
			  int end = idx-1;
			  for(int j=predict.entities.size()-1;j>=0;j--) {

				  if(predict.entities.get(j).end>=begin && predict.entities.get(j).end<=end)
					  return predict.entities.get(j);
			  }
			  return null;

		  }

	  }
	  
}

class Evaluate {
	double loss;
	double correct;
}
