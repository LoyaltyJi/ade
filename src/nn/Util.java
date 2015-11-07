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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.nlp.Segment;
import cn.fox.nlp.Sentence;
import cn.fox.nlp.TokenizerWithSegment;
import cn.fox.stanford.Tokenizer;
import cn.fox.utils.ObjectSerializer;
import cn.fox.utils.WordNetUtil;
import drug_side_effect_utils.Entity;
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
