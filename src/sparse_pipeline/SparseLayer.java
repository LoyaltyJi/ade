package sparse_pipeline;

import java.io.Serializable;
import java.util.List;
import java.util.Random;
import cn.fox.math.Function;



public class SparseLayer implements Serializable {
	public Parameters parameters;
	
	double[][] W;
	double[][] eg2W;
	
	boolean useB;
	double[] B;
	double[] eg2B;
 	
	public Father owner;
	int outputSize;
	
	public boolean debug;
	
	public SparseLayer(Parameters parameters, Father owner, int featureNumber, int outputSize, boolean debug, boolean useB) {
		this.parameters = parameters;
		this.owner = owner;
		this.outputSize = outputSize;
		this.debug = debug;
		this.useB = useB;
		
		Random random = new Random(System.currentTimeMillis());
		W = new double[outputSize][featureNumber];
		eg2W = new double[W.length][W[0].length];
		
		for(int i=0;i<W.length;i++) {
			for(int j=0;j<W[0].length;j++) {
				W[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
				//W[i][j] = 0;
			}
		}
		
		if(useB) {
			B = new double[outputSize];
			eg2B = new double[B.length];
			
			for(int i=0;i<B.length;i++) {
				B[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
	}
	
	public int giveTheBestChoice(Example ex) throws Exception {
		double[] scores = computeScores(ex);
		int optLabel = -1; // the label with the highest score
        for (int i = 0; i < outputSize; ++i) {
            if (optLabel < 0 || scores[i] > scores[optLabel])
              optLabel = i;  
        }
        return optLabel;
	}
	
	public double[] computeScores(Example ex) throws Exception {
		
		double[] scores = new double[outputSize];
		
		for(int j=0;j<scores.length;j++) {
			// W*X
			for(int i=0;i<ex.featureIdx.size();i++) {
				if(ex.featureIdx.get(i) == -1)
					continue;
				scores[j] += W[j][ex.featureIdx.get(i)];
			}
			// +B
			if(useB)
				scores[j] += B[j];
		}
		

        return scores;
	}
	
	public GradientKeeper process(List<Example> examples) throws Exception {
		GradientKeeper keeper = new GradientKeeper(parameters, this);
		double loss = 0;
		double correct = 0;

		// mini-batch
		for (Example ex : examples) {
			
			double[] scores = new double[outputSize];
			int optLabel = -1;
			for(int i=0;i<scores.length;i++) {
				// W*X
				for(int j=0;j<ex.featureIdx.size();j++) {
					if(ex.featureIdx.get(j)== -1)
						continue;
					scores[i] += W[i][ex.featureIdx.get(j)];
				}
				
				if(useB)
					scores[i] += B[i];
								
				if (optLabel < 0 || scores[i] > scores[optLabel])
		              optLabel = i;
			}

			// softmax
	        double sum1 = 0.0;
	        double sum2 = 0.0;
	        double maxScore = scores[optLabel];
	        for (int i = 0; i < outputSize; ++i) {
	            scores[i] = Function.exp(scores[i] - maxScore);
	            if (ex.label[i] == 1) sum1 += scores[i];
	            sum2 += scores[i];
	        }

	        // compute loss and correct rate of labels
	        loss += (Math.log(sum2) - Math.log(sum1)) / examples.size();
	        if (ex.label[optLabel] == 1)
	          correct += 1.0 / examples.size();
	        
	        
	        for(int i=0;i<outputSize;i++) {
	        	 double temp = -(ex.label[i] - scores[i] / sum2) / examples.size();
	        	 for(int j=0;j<ex.featureIdx.size();j++) {
	        		 if(ex.featureIdx.get(j) == -1)
	        			 continue;
	        		 keeper.gradW[i][ex.featureIdx.get(j)] += temp;
	        	 }
	        	 if(useB)
	        		 keeper.gradB[i] += temp;
	        }
 
		}
			
        // L2 Regularization
	    for (int i = 0; i < keeper.gradW.length; ++i) {
	        for (int j = 0; j < keeper.gradW[i].length; ++j) {
	          loss += parameters.regParameter * keeper.gradW[i][j] * keeper.gradW[i][j] / 2.0;
	          keeper.gradW[i][j] += parameters.regParameter * keeper.gradW[i][j];
	        }
	      }
	    
	    if(useB) {
	    	for(int i=0;i<keeper.gradB.length;i++) {
	    		loss += parameters.regParameter * keeper.gradB[i] * keeper.gradB[i] / 2.0;
	    		keeper.gradB[i] += parameters.regParameter * keeper.gradB[i];
	    	}
	    }
	    
	    
		
		if(debug)
			System.out.println("Cost = " + loss + ", Correct(%) = " + correct);
		
		return keeper;
  
	}
	
	public void updateWeights(GradientKeeper keeper) {
		for (int i = 0; i < W.length; ++i) {
	      for (int j = 0; j < W[i].length; ++j) {
	        eg2W[i][j] += keeper.gradW[i][j] * keeper.gradW[i][j];
	        W[i][j] -= parameters.adaAlpha * keeper.gradW[i][j] / Math.sqrt(eg2W[i][j] + parameters.adaEps);
	      }
	    }
		
		if(useB) {
	    	for(int i=0;i<keeper.gradB.length;i++) {
	    		eg2B[i] += keeper.gradB[i] * keeper.gradB[i] / 2.0;
	    		B[i] -= parameters.adaAlpha * keeper.gradB[i] / Math.sqrt(eg2B[i] + parameters.adaEps);
	    	}
	    }

	    
	}
	
	public void clearHistory() {
		eg2W = new double[W.length][W[0].length];
		if(useB)
			eg2B = new double[B.length];
	}
	
	public double computeLoss(Example ex) throws Exception {
		
		double[] scores = new double[outputSize];
		int optLabel = -1;
		for(int i=0;i<scores.length;i++) {
			// W*X
			for(int j=0;j<ex.featureIdx.size();j++) {
				if(ex.featureIdx.get(j) == -1)
					continue;
				scores[i] += W[i][ex.featureIdx.get(j)];
			}
							
			if (optLabel < 0 || scores[i] > scores[optLabel])
	              optLabel = i;
		}

		// softmax
        double sum1 = 0.0;
        double sum2 = 0.0;
        double maxScore = scores[optLabel];
        for (int i = 0; i < outputSize; ++i) {
            scores[i] = Function.exp(scores[i] - maxScore);
            if (ex.label[i] == 1) sum1 += scores[i];
            sum2 += scores[i];
        }

        // compute loss and correct rate of labels
        double loss = (Math.log(sum2) - Math.log(sum1)) ;
        
        
        return loss;
	}

	public void checkGradients (GradientKeeper keeper, List<Example> examples) throws Exception {
		// we only check Whs, Wbs and Wo
		System.out.println(Parameters.SEPARATOR+" Gradient checking begin");
		
		// randomly select one point
		Random random = new Random(System.currentTimeMillis());
		double epsilonGradientCheck = 1e-4;
		
		// Whs
		int check_i = random.nextInt(W.length);
		int check_j = random.nextInt(W[0].length);
		double orginValue = W[check_i][check_j];
		
		W[check_i][check_j] = orginValue + epsilonGradientCheck;
		double lossAdd = 0.0;
	  for (int i = 0; i < examples.size(); i++) {
	    Example oneExam = examples.get(i);
	    lossAdd += computeLoss(oneExam);
	  }

	  W[check_i][check_j] = orginValue - epsilonGradientCheck;
	  double lossSub = 0.0;
	  for (int i = 0; i < examples.size(); i++) {
	    Example oneExam = examples.get(i);
	    lossSub += computeLoss(oneExam);
	  }

	  double mockGrad = (lossAdd - lossSub) / (epsilonGradientCheck * 2);
	  mockGrad = mockGrad / examples.size();
	  double computeGrad = keeper.gradW[check_i][check_j];

	  System.out.printf("W[%d][%d] abs(mockGrad-computeGrad)= %.18f\n", check_i, check_j, Math.abs(mockGrad-computeGrad));
	  
	  // restore the value, important!!!
	  W[check_i][check_j] =  orginValue;

		System.out.println(Parameters.SEPARATOR+" Gradient checking end");
	}
	
	
}
