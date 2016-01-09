package sparse_pipeline;


import java.io.Serializable;
import java.util.Properties;
import edu.stanford.nlp.util.PropertiesUtils;


public class Parameters implements Serializable{
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
	
	  public double initRange = 0.01;
	  public int wordCutOff = 1;
	  public int maxIter = 1000;
	
	  public double adaEps = 1e-6;
	
	  public double adaAlpha = 0.01;
	
	  public double regParameter = 1e-8;
	 	
	  public int evalPerIter = 100;
	  
	  
	  public int batchSize = 50;

	  public Parameters(Properties properties) {
	    setProperties(properties);
	  }
	
	  private void setProperties(Properties props) {
		  initRange = PropertiesUtils.getDouble(props, "initRange", initRange);
		wordCutOff = PropertiesUtils.getInt(props, "wordCutOff", wordCutOff);
		maxIter = PropertiesUtils.getInt(props, "maxIter", maxIter);
		adaEps = PropertiesUtils.getDouble(props, "adaEps", adaEps);
		adaAlpha = PropertiesUtils.getDouble(props, "adaAlpha", adaAlpha);
		regParameter = PropertiesUtils.getDouble(props, "regParameter", regParameter);
		evalPerIter = PropertiesUtils.getInt(props, "evalPerIter", evalPerIter);
		batchSize = PropertiesUtils.getInt(props, "batchSize", batchSize);
		
		

	  }
	
	 	
	  public void printParameters() {
		  System.out.printf("initRange = %.2g%n", initRange);
		System.out.printf("wordCutOff = %d%n", wordCutOff);
		System.out.printf("maxIter = %d%n", maxIter);
		System.out.printf("adaEps = %.2g%n", adaEps);
		System.out.printf("adaAlpha = %.2g%n", adaAlpha);
		System.out.printf("regParameter = %.2g%n", regParameter);
		System.out.printf("evalPerIter = %d%n", evalPerIter);
		
		System.out.printf("batchSize = %d%n", batchSize);

		System.out.println(SEPARATOR);
	  }
}
