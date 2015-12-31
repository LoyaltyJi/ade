package nn;


import java.io.Serializable;
import java.util.Properties;
import edu.stanford.nlp.util.PropertiesUtils;


public class CombineParameters implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = -8092716117880301475L;
	/**
	  *   Out-of-vocabulary token string.
	  */
	  public static final String UNKNOWN = "-UNKN-";
	  // Out of window token string
	  public static final String PADDING = "-PAD-";
	  public static final String SEPARATOR = "###################";
	  public static final String CHEMICAL = "Chemical";
	  public static final String DISEASE = "Disease";
	  public static final String RELATION = "CID";
	
	  /**
	   * Refuse to train on words which have a corpus frequency less than
	   * this number.
	   */
	  public int wordCutOff1 = 2;
	  public int wordCutOff2 = 1;
	
	  /**
	   * Model weights will be initialized to random values within the
	   * range {@code [-initRange, initRange]}.
	   */
	  public double initRange = 0.01;
	
	  /**
	   * Maximum number of iterations for training
	   */
	  public int maxIter = 1000;
	
	  	
	  /**
	   * An epsilon value added to the denominator of the AdaGrad
	   * expression for numerical stability
	   */
	  public double adaEps = 1e-6;
	
	  /**
	   * Initial global learning rate for AdaGrad training
	   */
	  public double adaAlpha = 0.01;
	
	  /**
	   * Regularization parameter. All weight updates are scaled by this
	   * single parameter.
	   */
	  public double regParameter = 1e-8;
	
	  /**
	   * Dropout probability. For each training example we randomly choose
	   * some amount of units to disable in the neural network classifier.
	   * This probability controls the proportion of units "dropped out."
	   */
	  public double dropProb = 0.5;
	
	  
	
	  /**
	   * Dimensionality of the word embeddings used
	   */
	  public int embeddingSize = 50;
	
	  /**
	   * Number which we should keep into precompute.
	   * Both word and position are considered. For examples, if we only have word features and
	   * the number of words is 5000, and the model has 10 embedding features(excluding the composite
	   * features), it will be enough to set this value as 50000. Nevertheless, if you don't want to 
	   * precompute all, you can set a smaller value, e.g., 25000.
	   */
	  public int numPreComputed = 70000;
	
	  /**
	   * During training, run a full evaluation after every
	   * {@code evalPerIter} iterations.
	   */
	  public int evalPerIter = 100;
	  
	  
	  /**
	   * The parameters below is about the architecture of NN.
	   */
	  public int lossFunction = 1; // 1-CrossEntropy; 
	  
	  public int actFuncOfOutput = 1; // 1-Softmax; 
	  
	  public int actFuncOfHidden = 1; // 1:cube 2:tanh 3:relu
		
	  public int hiddenLevel = 1;
	  public int hiddenSize = 200;
	  
	  // The size of context windows
	  public int windowSize = 3;
	  
	
	  
	  // other, newChemical, newDisease, append, notConnect, connect  
	  // start at 0
	  public int outputSize = 6;
	  
	  
	  /**
	   * Size of mini-batch for training. A random subset of training
	   * examples of this size will be used to train the classifier on each
	   * iteration.
	   * Instead of a fixed number, we use the percentage because the numbers of 
	   * entity examples and relation ones are not balanced.
	   */
	  public double batchEntityPercent = 0.1;
	  public double batchRelationPercent = 0.1;
	  
	  // for gradient checking
	  public double epsilonGradientCheck = 1e-4;

	
	  // 1 - CaseSensitive, 2-lemma, 3-pattern, else lowercase
	  // if this one has changed, the embedding file should also change.
	  public int wordPreprocess = 0;
	  
	  public int prefixLength = 4;
	  
	  // the output node number of the filter
	  public int entityDimension =  50;
	  public boolean entityConvolution = false;
	  public int entityPooling = 1; // 1-avg, 2-max
	  
	  public int sentenceDimension = 50;
	  public boolean sentenceConvolution = false;
	  
	  public int positionDimension = 50;
	  public boolean usePosition = false;
	  
	  // at least 1
	  public int beamSize = 1;
	  		
	  public CombineParameters(Properties properties) {
	    setProperties(properties);
	  }
	
	  private void setProperties(Properties props) {
		wordCutOff1 = PropertiesUtils.getInt(props, "wordCutOff1", wordCutOff1);
		wordCutOff2 = PropertiesUtils.getInt(props, "wordCutOff2", wordCutOff2);
		initRange = PropertiesUtils.getDouble(props, "initRange", initRange);
		maxIter = PropertiesUtils.getInt(props, "maxIter", maxIter);
		adaEps = PropertiesUtils.getDouble(props, "adaEps", adaEps);
		adaAlpha = PropertiesUtils.getDouble(props, "adaAlpha", adaAlpha);
		regParameter = PropertiesUtils.getDouble(props, "regParameter", regParameter);
		dropProb = PropertiesUtils.getDouble(props, "dropProb", dropProb);
		embeddingSize = PropertiesUtils.getInt(props, "embeddingSize", embeddingSize);
		numPreComputed = PropertiesUtils.getInt(props, "numPreComputed", numPreComputed);
		evalPerIter = PropertiesUtils.getInt(props, "evalPerIter", evalPerIter);
		
		lossFunction = PropertiesUtils.getInt(props, "lossFunction", lossFunction);
		actFuncOfOutput = PropertiesUtils.getInt(props, "actFuncOfOutput", actFuncOfOutput);
		actFuncOfHidden = PropertiesUtils.getInt(props, "actFuncOfHidden", actFuncOfHidden);
		hiddenLevel = PropertiesUtils.getInt(props, "hiddenLevel", hiddenLevel);
		hiddenSize = PropertiesUtils.getInt(props, "hiddenSize", hiddenSize);
		windowSize = PropertiesUtils.getInt(props, "windowSize", windowSize);
		outputSize = PropertiesUtils.getInt(props, "outputSize", outputSize);
		
		
		batchEntityPercent = PropertiesUtils.getDouble(props, "batchEntityPercent", batchEntityPercent);
		batchRelationPercent = PropertiesUtils.getDouble(props, "batchRelationPercent", batchRelationPercent);
		epsilonGradientCheck = PropertiesUtils.getDouble(props, "epsilonGradientCheck", epsilonGradientCheck);
		wordPreprocess = PropertiesUtils.getInt(props, "wordPreprocess", wordPreprocess);
		prefixLength = PropertiesUtils.getInt(props, "prefixLength", prefixLength);
		
		entityDimension = PropertiesUtils.getInt(props, "entityDimension", entityDimension);
		entityConvolution = PropertiesUtils.getBool(props, "entityConvolution", entityConvolution);
		entityPooling = PropertiesUtils.getInt(props, "entityPooling", entityPooling);
		
		if(entityConvolution == false && entityDimension!=embeddingSize) {
			System.out.println("if use pooling only(entityConvolution=false), entityDimension must be equal to embeddingSize");
			System.exit(0);
		}
		
		sentenceDimension = PropertiesUtils.getInt(props, "sentenceDimension", sentenceDimension);
		sentenceConvolution = PropertiesUtils.getBool(props, "sentenceConvolution", sentenceConvolution);
		
		positionDimension = PropertiesUtils.getInt(props, "positionDimension", positionDimension);
		usePosition = PropertiesUtils.getBool(props, "usePosition", usePosition);
			
		
		beamSize = PropertiesUtils.getInt(props, "beamSize", beamSize);
	  }
	
	 	
	  public void printParameters() {
		System.out.printf("wordCutOff1 = %d%n", wordCutOff1);
		System.out.printf("wordCutOff2 = %d%n", wordCutOff2);
		System.out.printf("initRange = %.2g%n", initRange);
		System.out.printf("maxIter = %d%n", maxIter);
		System.out.printf("adaEps = %.2g%n", adaEps);
		System.out.printf("adaAlpha = %.2g%n", adaAlpha);
		System.out.printf("regParameter = %.2g%n", regParameter);
		System.out.printf("dropProb = %.2g%n", dropProb);
		System.out.printf("embeddingSize = %d%n", embeddingSize);
		System.out.printf("numPreComputed = %d%n", numPreComputed);
		System.out.printf("evalPerIter = %d%n", evalPerIter);
		
		System.out.printf("lossFunction = %d%n", lossFunction);
		System.out.printf("actFuncOfOutput = %d%n", actFuncOfOutput);
		System.out.printf("actFuncOfHidden = %d%n", actFuncOfHidden);
		System.out.printf("hiddenLevel = %d%n", hiddenLevel);
		System.out.printf("hiddenSize = %d%n", hiddenSize);
		System.out.printf("windowSize = %d%n", windowSize);
		System.out.printf("outputSize = %d%n", outputSize);
		
		
		System.out.printf("batchEntityPercent = %.2g%n", batchEntityPercent);
		System.out.printf("batchRelationPercent = %.2g%n", batchRelationPercent);
		System.out.printf("epsilonGradientCheck = %.2g%n", epsilonGradientCheck);
		System.out.printf("wordPreprocess = %d%n", wordPreprocess);
		System.out.printf("prefixLength = %d%n", prefixLength);
		
		System.out.printf("entityDimension = %d%n", entityDimension);
		System.out.printf("entityConvolution = %b%n", entityConvolution);
		System.out.printf("entityPooling = %d%n", entityPooling);
		
		System.out.printf("sentenceDimension = %d%n", sentenceDimension);
		System.out.printf("sentenceConvolution = %b%n", sentenceConvolution);
		
		System.out.printf("positionDimension = %d%n", positionDimension);
		System.out.printf("usePosition = %b%n", usePosition);
		
		System.out.printf("beamSize = %d%n", beamSize);
		
		System.out.println(SEPARATOR);
	  }
}
