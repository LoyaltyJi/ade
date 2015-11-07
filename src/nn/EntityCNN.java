package nn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import gnu.trove.TIntArrayList;


public class EntityCNN implements Serializable {

	private static final long serialVersionUID = -6471262603387658363L;
	
	public Parameters parameters;
	
	public NNADE nnade;
	public NN nn;
	
	public double[][] filterW;
	public double[] filterB;
	public int filterWindow = 2;
	
	public double[][] eg2filterW;
	public double[] eg2filterB;
	
	public boolean debug;
	
	public EntityCNN(Parameters parameters, NNADE nnade, NN nn, boolean debug) {
		this.parameters = parameters;
		this.nnade = nnade;
		this.nn = nn;
		this.debug = debug;
		
		Random random = new Random(System.currentTimeMillis());
		
		filterW = new double[parameters.entityDimension][filterWindow*parameters.embeddingSize];
		eg2filterW = new double[filterW.length][filterW[0].length];
		for(int i=0;i<filterW.length;i++) {
			for(int j=0;j<filterW[0].length;j++) {
				filterW[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		
		filterB = new double[parameters.entityDimension];
		eg2filterB = new double[filterB.length];
		for(int i=0;i<filterB.length;i++) {
			filterB[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
		}

		
	}
	
	public State forward(TIntArrayList entityIdx) throws Exception {
				
		// input -> convolutional map
		int timeToConvol = entityIdx.size()-filterWindow+1; // the times to convolute
		if(timeToConvol<=0)
			timeToConvol = 1; // sentence is less than window
		
		double[][] S = new double[timeToConvol][parameters.entityDimension]; // S[k][j]
		for(int k=0;k<timeToConvol;k++) { // k corresponds the begin position of each convolution
			int offset = 0;
			for(int wordCount=0;wordCount<filterWindow;wordCount++) {
				int wordIdx = k+wordCount;
				double[] emb = null;
				if(wordIdx<=entityIdx.size()-1) {
					emb = nnade.E[entityIdx.get(wordIdx)];
				} else {
					emb = nnade.E[nnade.getPaddingID()];
				}
				
				for(int j=0;j<parameters.entityDimension;j++) {
					for(int m=0;m<parameters.embeddingSize;m++) {
						S[k][j] += filterW[j][offset+m]*emb[m]; // W*Z
					}
				}
				offset += parameters.embeddingSize;
			}
			
			
			for(int j=0;j<parameters.entityDimension;j++) {
				S[k][j] += filterB[j];  // W*Z+B
				S[k][j] = Util.sigmoid(S[k][j]); // activation
			}
			
			

		}
		
		// S has been done, begin max-pooling to generate X
		double[] X = new double[parameters.entityDimension];
		int[] maxRemember = new int[X.length];
		
		for(int j=0;j<X.length;j++) {
			double max = S[0][j];
			int maxK = 0;
			for(int k=1;k<S.length;k++) {
				if(S[k][j]>max) {
					max = S[k][j];
					maxK = k;
				}
			}
			X[j] = max;

			maxRemember[j] = maxK;
		}
				
		State state = new State();
		state.emb = X;
		state.maxRemember = maxRemember;
		state.S = S;
		return state;
	}
	
	public void backward(double[] gradX, TIntArrayList entityIdx, State state, EntityCNNGradientKeeper keeper) throws Exception {
		
		
		// X -> S		
		double[][] gradS = new double[state.S.length][state.S[0].length];
		int[] maxRemember = state.maxRemember;
		for(int j=0;j<gradS[0].length;j++) {
			for(int k=0;k<gradS.length;k++) {
				if(maxRemember[j]==k)
					gradS[k][j] = gradX[j];
				else
					gradS[k][j] = 0;
			}
		}
		
		// S -> Z
		double[][] S = state.S;
		for(int k=0;k<gradS.length;k++) {
			
			int offset = 0;
			for(int wordCount=0;wordCount<filterWindow;wordCount++) {
				int wordIdx = k+wordCount;
				double[] emb = null;
				int embId = -1;
				if(wordIdx<=entityIdx.size()-1) {
					embId = entityIdx.get(wordIdx);
				} else {
					embId = nnade.getPaddingID();
				}
				emb = nnade.E[embId];
				
				for(int j=0;j<gradS[0].length;j++) {
					double delta2 = gradS[k][j]*S[k][j]*(1-S[k][j]);
					for(int m=0;m<parameters.embeddingSize;m++) {
						keeper.gradW[j][offset+m] += delta2*emb[m];
						if(parameters.bEmbeddingFineTune)
							keeper.gradE[embId][m] += delta2*filterW[j][offset+m];
					}
					keeper.gradB[j] += delta2;
				}
				offset += parameters.embeddingSize;
			}
		
		}
		
		// L2 Regularization
	    for (int i = 0; i < keeper.gradW.length; ++i) {
	        for (int j = 0; j < keeper.gradW[i].length; ++j) {
	        	keeper.gradW[i][j] += parameters.regParameter * filterW[i][j];
	        }
	      }

		for(int i=0; i < keeper.gradB.length; i++) {
			keeper.gradB[i] += parameters.regParameter * filterB[i];
		}
		
		if(parameters.bEmbeddingFineTune) {
			for(int i=0; i< keeper.gradE.length; i++) {
				for(int j=0; j < keeper.gradE[0].length;j++) {
					keeper.gradE[i][j] += parameters.regParameter * nnade.E[i][j];
				}
			}
		}

	}
	
	
	public void updateWeights(EntityCNNGradientKeeper keeper) {
		// ada-gradient
	    for (int i = 0; i < filterW.length; ++i) {
	        for (int j = 0; j < filterW[i].length; ++j) {
	          eg2filterW[i][j] += keeper.gradW[i][j] * keeper.gradW[i][j];
	          filterW[i][j] -= parameters.adaAlpha * keeper.gradW[i][j] / Math.sqrt(eg2filterW[i][j] + parameters.adaEps);
	        }
	      }

		for(int i=0; i < filterB.length; i++) {
			eg2filterB[i] += keeper.gradB[i] * keeper.gradB[i];
			filterB[i] -= parameters.adaAlpha * keeper.gradB[i] / Math.sqrt(eg2filterB[i] + parameters.adaEps); 
		}

		
		// here we don't update E, but give the job to NN.updateWeights
		// because their gradE are same

		
	}

}

class EntityCNNGradientKeeper {
	public double[][] gradW;
	public double[] gradB;
	public double[][] gradE;
	
	
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public EntityCNNGradientKeeper(Parameters parameters, EntityCNN cnn, GradientKeeper gk) {
		gradW = new double[cnn.filterW.length][cnn.filterW[0].length];
		gradB = new double[cnn.filterB.length];
		if(parameters.bEmbeddingFineTune)
			gradE = gk.gradE;
			
	}
}
