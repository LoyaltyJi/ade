package nn;

import java.io.Serializable;
import java.util.Random;

import cn.fox.math.Function;
import gnu.trove.TIntArrayList;

public class SentenceCNN implements Serializable {
	private static final long serialVersionUID = 5434993506255607840L;
	
	public Parameters parameters;
	
	public Father nnade;
	
	public double[][] filterW;
	public double[] filterB;
	public int filterWindow = 2;
	
	public double[][] eg2filterW;
	public double[] eg2filterB;
	
	public boolean debug;
	

	public SentenceCNN(Parameters parameters, Father nnade, boolean debug) {
		this.parameters = parameters;
		this.nnade = nnade;
		this.debug = debug;
		
		Random random = new Random(System.currentTimeMillis());
		
		if(parameters.usePosition) // window*(WF+PF1+PF2)
			filterW = new double[parameters.sentenceDimension][filterWindow*parameters.embeddingSize*3]; 
		else
			filterW = new double[parameters.sentenceDimension][filterWindow*parameters.embeddingSize];
		
		eg2filterW = new double[filterW.length][filterW[0].length];
		for(int i=0;i<filterW.length;i++) {
			for(int j=0;j<filterW[0].length;j++) {
				filterW[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		
		filterB = new double[parameters.sentenceDimension];
		eg2filterB = new double[filterB.length];
		for(int i=0;i<filterB.length;i++) {
			filterB[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
		}

		
	}
	
	public State forward(Example example) throws Exception {
		
		// input -> convolutional map
		int timeToConvol = example.sentenceIdx.size()-filterWindow+1; // the times to convolute
		if(timeToConvol<=0)
			timeToConvol = 1; // sentence is less than window
		
		double[][] S = new double[timeToConvol][parameters.sentenceDimension]; // S[k][j]
		for(int k=0;k<timeToConvol;k++) { // k corresponds the begin position of each convolution
			int offset = 0;
			for(int wordCount=0;wordCount<filterWindow;wordCount++) {
				int wordIdx = k+wordCount;
				double[] emb = null;
				if(wordIdx<=example.sentenceIdx.size()-1) {
					emb = nnade.getE()[example.sentenceIdx.get(wordIdx)];
				} else {
					emb = nnade.getE()[nnade.getPaddingID()];
				}
				
				for(int j=0;j<parameters.sentenceDimension;j++) {
					for(int m=0;m<parameters.embeddingSize;m++) {
						S[k][j] += filterW[j][offset+m]*emb[m]; // W*Z
					}
				}
				offset += parameters.embeddingSize;
				
				if(parameters.usePosition) {
					if(wordIdx<=example.positionIdxFormer.size()-1) {
						emb = nnade.getE()[example.positionIdxFormer.get(wordIdx)];
					} else {
						emb = nnade.getE()[nnade.getPositionID(0)];
					}
					for(int j=0;j<parameters.sentenceDimension;j++) {
						for(int m=0;m<parameters.embeddingSize;m++) {
							S[k][j] += filterW[j][offset+m]*emb[m]; // W*Z
						}
					}
					offset += parameters.embeddingSize;
					
					if(wordIdx<=example.positionIdxLatter.size()-1) {
						emb = nnade.getE()[example.positionIdxLatter.get(wordIdx)];
					} else {
						emb = nnade.getE()[nnade.getPositionID(0)];
					}
					for(int j=0;j<parameters.sentenceDimension;j++) {
						for(int m=0;m<parameters.embeddingSize;m++) {
							S[k][j] += filterW[j][offset+m]*emb[m]; // W*Z
						}
					}
					offset += parameters.embeddingSize;
					
				} 
			}
			
			
			for(int j=0;j<parameters.sentenceDimension;j++) {
				S[k][j] += filterB[j];  // W*Z+B
				//S[k][j] = Function.sigmoid(S[k][j]); // activation
				S[k][j] = Function.tanh(S[k][j]);
			}
			
			

		}
		
		// S has been done, begin max-pooling to generate X
		double[] X = new double[parameters.sentenceDimension];
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
	
	public void backward(double[] gradX, Example example, State state, SentenceCNNGradientKeeper keeper) throws Exception {
		
		
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
				if(wordIdx<=example.sentenceIdx.size()-1) {
					embId = example.sentenceIdx.get(wordIdx);
				} else {
					embId = nnade.getPaddingID();
				}
				emb = nnade.getE()[embId];
				
				for(int j=0;j<gradS[0].length;j++) {
					//double delta2 = gradS[k][j]*S[k][j]*(1-S[k][j]);
					double delta2 = gradS[k][j]*Function.deriTanh(S[k][j]);
					for(int m=0;m<parameters.embeddingSize;m++) {
						keeper.gradW[j][offset+m] += delta2*emb[m];
						if(parameters.bEmbeddingFineTune)
							keeper.gradE[embId][m] += delta2*filterW[j][offset+m];
					}
					keeper.gradB[j] += delta2;
				}
				offset += parameters.embeddingSize;
				
				if(parameters.usePosition) {
					if(wordIdx<=example.positionIdxFormer.size()-1) {
						embId = example.positionIdxFormer.get(wordIdx);
					} else {
						embId = nnade.getPositionID(0);
					}
					emb = nnade.getE()[embId];
					
					for(int j=0;j<gradS[0].length;j++) {
						//double delta2 = gradS[k][j]*S[k][j]*(1-S[k][j]);
						double delta2 = gradS[k][j]*Function.deriTanh(S[k][j]);
						for(int m=0;m<parameters.embeddingSize;m++) {
							keeper.gradW[j][offset+m] += delta2*emb[m];
							if(parameters.bEmbeddingFineTune)
								keeper.gradE[embId][m] += delta2*filterW[j][offset+m];
						}
						keeper.gradB[j] += delta2;
					}
					offset += parameters.embeddingSize;
					
					if(wordIdx<=example.positionIdxLatter.size()-1) {
						embId = example.positionIdxLatter.get(wordIdx);
					} else {
						embId = nnade.getPositionID(0);
					}
					emb = nnade.getE()[embId];
					
					for(int j=0;j<gradS[0].length;j++) {
						//double delta2 = gradS[k][j]*S[k][j]*(1-S[k][j]);
						double delta2 = gradS[k][j]*Function.deriTanh(S[k][j]);
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
					keeper.gradE[i][j] += parameters.regParameter * nnade.getE()[i][j];
				}
			}
		}

	}
	
	public void updateWeights(SentenceCNNGradientKeeper keeper) {
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

class SentenceCNNGradientKeeper {
	public double[][] gradW;
	public double[] gradB;
	public double[][] gradE;
	
	
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public SentenceCNNGradientKeeper(Parameters parameters, SentenceCNN cnn, GradientKeeper gk) {
		gradW = new double[cnn.filterW.length][cnn.filterW[0].length];
		gradB = new double[cnn.filterB.length];
		if(parameters.bEmbeddingFineTune)
			gradE = gk.gradE;
			
	}
	
	public SentenceCNNGradientKeeper(Parameters parameters, SentenceCNN cnn, GradientKeeper1 gk) {
		gradW = new double[cnn.filterW.length][cnn.filterW[0].length];
		gradB = new double[cnn.filterB.length];
		if(parameters.bEmbeddingFineTune)
			gradE = gk.gradE;
			
	}
}
