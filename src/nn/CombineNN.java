package nn;

import static java.util.stream.Collectors.toSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import cc.mallet.types.SparseVector;
import cn.fox.math.Function;
import gnu.trove.TIntArrayList;


// It's a simple version of NN, because there are no multi-hidden layers.
public class CombineNN implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1908284566872396134L;
	public CombineParameters parameters;
	/*
	 *  the weights of the hidden layers
	 *  for first hidden layer, hiddenSize x inputSize
	 *  for other hidden layers, hiddenSize x hiddenSize
	 */
	public double[][] Wh;
	// the bias of the hidden layers, hiddenSize x 1
	public double[] Bh;
	// the weights of the output layer, outputSize x hiddenSize
	public double[][] Wo;
	
	/*
	 * precompute
	 * key - the row id in E
	 * value - the row id in saved
	 */
	public final HashMap<Integer, Integer> preMap;
	// row - value in preMap
	// column - the hidden layer index (hiddenSize)
	public double[][] saved;
	public double[][] gradSaved;
	
	// Gradient histories used by AdaGradient
	public double[][] eg2Wh; 
	public double[] eg2Bh;
	public double[][] eg2Wo;
	
	
	public boolean debug;
	
	public Combine owner;
	
	public int embFeatureNumber;
	
	public CombineEntityCNN entityCNN;
	
		
	public CombineNN(CombineParameters parameters, Combine owner, TIntArrayList preComputed, Example example) {
		super();
		this.parameters = parameters;
		this.owner = owner;
		this.embFeatureNumber = example.featureIdx.size();
		
		int outputSize = parameters.outputSize;
		// decide the input based on the example
		int inputSize = embFeatureNumber*parameters.embeddingSize+parameters.entityDimension*2;
		if(parameters.sentenceConvolution)
			inputSize += parameters.sentenceDimension;
		
		Random random = new Random(System.currentTimeMillis());
		
		Wo = new double[outputSize][parameters.hiddenSize];
		eg2Wo = new double[Wo.length][Wo[0].length];
		for(int i=0;i<Wo.length;i++) {
			for(int j=0;j<Wo[0].length;j++) {
				Wo[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		
		Wh = new double[parameters.hiddenSize][inputSize];
		eg2Wh = new double[Wh.length][Wh[0].length];
		for(int i=0;i<Wh.length;i++) {
			for(int j=0;j<Wh[0].length;j++) {
				Wh[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		
		Bh = new double[parameters.hiddenSize];
		eg2Bh = new double[Bh.length];
		for(int i=0;i<Bh.length;i++) {
			Bh[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
		}
		
		preMap = new HashMap<>();
	    for (int i = 0; i < preComputed.size(); ++i)
	      preMap.put(preComputed.get(i), i);
			    
	    if(parameters.entityConvolution)
	    	entityCNN = new CombineEntityCNN(parameters, owner, debug);
	    

	}
	
	public int giveTheBestChoice(Example ex) throws Exception {
		double[] scores = computeScores(ex);
		int optLabel = -1; // the label with the highest score
        for (int i = 0; i < parameters.outputSize; ++i) {
            if (optLabel < 0 || scores[i] > scores[optLabel])
              optLabel = i;  
        }
        return optLabel;
	}
	
	
	
	public State generateEntityComposite(TIntArrayList entityIdx) throws Exception {
		if(parameters.entityConvolution) {
			return entityCNN.forward(entityIdx);
		} else {
			throw new Exception();
		}
	}
	
	/*
	 * Given a example, compute from bottom to up and get the output scores.
	 * This cannot be used in training, because we don't consider dropout.
	 */
	public double[] computeScores(Example ex) throws Exception {
		
		double[] scores = new double[parameters.outputSize];
		double[] hidden = new double[parameters.hiddenSize];
		
		int offset = 0;
        for (int j = 0; j < embFeatureNumber; ++j, offset += parameters.embeddingSize) {
          int tok = ex.featureIdx.get(j);
          if(tok==-1)
        	  continue;
          int index = tok * embFeatureNumber + j;
          if (preMap.containsKey(index)) {
            int id = preMap.get(index);
            for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++)
              hidden[nodeIndex] += saved[id][nodeIndex];
          } else {
        	  for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
        		          			  
                  for (int k = 0; k < parameters.embeddingSize; ++k)
                    hidden[nodeIndex] += Wh[nodeIndex][offset + k] * owner.getE()[tok][k];
              }
          }
          
          
                    
        }
        
        if(ex.bRelation) {
        	double[] formerEmb = generateEntityComposite(ex.formerIdx).emb;
        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
        		for (int k = 0; k < formerEmb.length; ++k)
        			hidden[nodeIndex] += Wh[nodeIndex][offset+k] * formerEmb[k];
        	}
        	offset += formerEmb.length;
        	
        	double[] latterEmb = generateEntityComposite(ex.latterIdx).emb;
        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
        		for (int k = 0; k < latterEmb.length; ++k)
        			hidden[nodeIndex] += Wh[nodeIndex][offset+k] * latterEmb[k];
        	}
        	offset += latterEmb.length;
        	
        	
        	
        }
        
        // add bias and activation
        double[] hidden3 = new double[parameters.hiddenSize];
        for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
            hidden[nodeIndex] += Bh[nodeIndex];
            if(parameters.actFuncOfHidden == 3) {
            	hidden3[nodeIndex] = Function.relu(hidden[nodeIndex]);
            } else if(parameters.actFuncOfHidden == 1) {
            	hidden3[nodeIndex] = Math.pow(hidden[nodeIndex], 3);
			} else if(parameters.actFuncOfHidden == 2) {
				hidden3[nodeIndex] = Function.tanh(hidden[nodeIndex]);
			} else {
				throw new Exception();
			}
            
        }

        // hidden -> output
		for (int i = 0; i < parameters.outputSize; ++i) {
        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++)
              scores[i] += Wo[i][nodeIndex] * hidden3[nodeIndex];

        }
		
		
        return scores;
	}
		
	/*
	 * Given some examples, compute from bottom to top for each example
	 * back-propagate their gradient
	 */
	public GradientKeeper1 process(List<Example> examples, Perceptron perceptron) throws Exception {
		// precompute
		Set<Integer> toPreCompute = getToPreCompute(examples);
	    preCompute(toPreCompute);
	    
	    gradSaved = new double[preMap.size()][parameters.hiddenSize];
			    
	    GradientKeeper1 keeper = new GradientKeeper1(parameters, this);
	    EntityCNNGradientKeeper entityKeeper = null;
	    if(parameters.entityConvolution)
	    	entityKeeper = new EntityCNNGradientKeeper(parameters, entityCNN, keeper);
	    keeper.entityKeeper = entityKeeper;
	    
		
	    double loss = 0;
		double correct = 0;
		Random random = new Random(System.currentTimeMillis());
		
		// mini-batch
		for (Example ex : examples) {
			
			double[] scores = new double[parameters.outputSize];
			// Run dropout: randomly drop some hidden-layer units. `ls`
	        // contains the indices of those units which are still active
			int [] ls = IntStream.range(0, parameters.hiddenSize)
	                            .filter(n-> random.nextDouble() > parameters.dropProb)
	                            .toArray();
			double[] hidden = new double[parameters.hiddenSize];
			
			 // input -> hidden1
	        int offset = 0;
	        for (int j = 0; j < embFeatureNumber; ++j, offset += parameters.embeddingSize) {
	          int tok = ex.featureIdx.get(j);
	          if(tok==-1)
	        	  continue;
	          int index = tok * embFeatureNumber + j;
	          if (preMap.containsKey(index)) {
	            int id = preMap.get(index);
	            for (int nodeIndex : ls)
	              hidden[nodeIndex] += saved[id][nodeIndex];
	          } else {
	        	  for (int nodeIndex : ls) {
		              for (int k = 0; k < parameters.embeddingSize; ++k)
		                hidden[nodeIndex] += Wh[nodeIndex][offset + k] * owner.getE()[tok][k];
		            }
	          }
	          
	          

	          
	          
	        }
	        
	        if(ex.bRelation) {
	        	double[] formerEmb = generateEntityComposite(ex.formerIdx).emb;
	        	for(int nodeIndex : ls) {
	        		for (int k = 0; k < formerEmb.length; ++k)
	        			hidden[nodeIndex] += Wh[nodeIndex][offset+k] * formerEmb[k];
	        	}
	        	offset += formerEmb.length;
	        	
	        	double[] latterEmb = generateEntityComposite(ex.latterIdx).emb;
	        	for(int nodeIndex : ls) {
	        		for (int k = 0; k < latterEmb.length; ++k)
	        			hidden[nodeIndex] += Wh[nodeIndex][offset+k] * latterEmb[k];
	        	}
	        	offset += latterEmb.length;
	        	
	        	
	        }
	        
	        // add bias and activation
	        double[] hidden3 = new double[parameters.hiddenSize];
	        for (int nodeIndex : ls) {
	            hidden[nodeIndex] += Bh[nodeIndex];
	            if(parameters.actFuncOfHidden == 3) {
	            	hidden3[nodeIndex] = Function.relu(hidden[nodeIndex]);
	            } else if(parameters.actFuncOfHidden == 1) {
	            	hidden3[nodeIndex] = Math.pow(hidden[nodeIndex], 3);
				} else if(parameters.actFuncOfHidden == 2) {
					hidden3[nodeIndex] = Function.tanh(hidden[nodeIndex]);
				} else {
					throw new Exception();
				}
	            
	        }
        			
			
	        // hidden -> output
			int optLabel = -1; // the label with the highest score
	        for (int i = 0; i < parameters.outputSize; ++i) {
	            for (int nodeIndex : ls)
	              scores[i] += Wo[i][nodeIndex] * hidden3[nodeIndex];

	            if (optLabel < 0 || scores[i] > scores[optLabel])
	              optLabel = i;
	          
	        }

	        double sum1 = 0.0;
	        double sum2 = 0.0;
	        double maxScore = scores[optLabel];
	        for (int i = 0; i < parameters.outputSize; ++i) {
	        	// softmax
	            scores[i] = Math.exp(scores[i] - maxScore);
	            if (ex.label[i] == 1) sum1 += scores[i];
	            sum2 += scores[i];
	          
	        }

	        // compute loss and correct rate of labels
	        loss += (Math.log(sum2) - Math.log(sum1)) / examples.size();
	        if (ex.label[optLabel] == 1)
	          correct += 1.0 / examples.size();
	        
			
			// Until now, the forward procedure has ended and backward begin
	        
	                
			// output -> highest hidden layer
	        double[] gradHidden3 = new double[parameters.hiddenSize];
	        for (int i = 0; i < parameters.outputSize; ++i) {
	          
	            double temp = -(ex.label[i] - scores[i] / sum2) / examples.size();
	            for (int nodeIndex : ls) {
	              keeper.gradWo[i][nodeIndex] += temp * hidden3[nodeIndex];
	              gradHidden3[nodeIndex] += temp * Wo[i][nodeIndex];
	            }
	        }
	        
	        double[] gradHidden = new double[parameters.hiddenSize];
	        for (int nodeIndex : ls) {
	        	if(parameters.actFuncOfHidden == 3) {
	        		gradHidden[nodeIndex] = gradHidden3[nodeIndex] * Function.deriRelu(hidden[nodeIndex]);
	            } else if(parameters.actFuncOfHidden == 1)
	        		gradHidden[nodeIndex] = gradHidden3[nodeIndex] * 3 * hidden[nodeIndex] * hidden[nodeIndex];
	        	else if(parameters.actFuncOfHidden == 2) {
	        		gradHidden[nodeIndex] = gradHidden3[nodeIndex] * Function.deriTanh(hidden3[nodeIndex]);
				} else
	        		throw new Exception();
	        	keeper.gradBh[nodeIndex] += gradHidden[nodeIndex]; // in chen 2014 EMNLP, this is gradHidden3 but I think it's wrong
	        }
	        

        	offset = 0;
	        for (int j = 0; j < embFeatureNumber; ++j, offset += parameters.embeddingSize) {
	          int tok = ex.featureIdx.get(j);
	          if(tok==-1)
	        	  continue;
	          int index = tok * embFeatureNumber + j;
	          if (preMap.containsKey(index)) {
	            int id = preMap.get(index);
	            for (int nodeIndex : ls)
	              gradSaved[id][nodeIndex] += gradHidden[nodeIndex];
	          } else {
	        	  for (int nodeIndex : ls) {
	        		  for (int k = 0; k < parameters.embeddingSize; ++k) {
							keeper.gradWh[nodeIndex][offset + k] += gradHidden[nodeIndex] * owner.getE()[tok][k];
	        		  }
	        		  
	        		  if(tok < owner.getKnownWords1().size() || 
	        				  tok >= owner.getKnownWords1().size()+owner.getKnownWords2().size())  {
	        			  for (int k = 0; k < parameters.embeddingSize; ++k) 
	        				  keeper.gradE[tok][k] += gradHidden[nodeIndex] * Wh[nodeIndex][offset + k];
	        		  } 
	        		  
	        	  }
	          }
				

	        }
	        
	        if(ex.bRelation) {
	        	
	        	State formerState = generateEntityComposite(ex.formerIdx);
	        	double[] formerEmb = formerState.emb;
	        	double[] formerError = new double[formerEmb.length];
	        	for(int nodeIndex : ls) {
	        		for (int k = 0; k < formerEmb.length; ++k) {
	        			keeper.gradWh[nodeIndex][offset + k] += gradHidden[nodeIndex] * formerEmb[k];
	        			formerError[k] += gradHidden[nodeIndex] * Wh[nodeIndex][offset+k];
	        		}
	        	}
	        	offset += formerEmb.length;
	        	if(parameters.entityConvolution) {
	        		// continue back-propagate to CNN
	        		entityCNN.backward(formerError, ex.formerIdx, formerState, entityKeeper);
	        	}
	        	
	        	State latterState = generateEntityComposite(ex.latterIdx);
	        	double[] latterEmb = latterState.emb;
	        	double[] latterError = new double[latterEmb.length];
	        	for(int nodeIndex : ls) {
	        		for (int k = 0; k < latterEmb.length; ++k) {
	        			keeper.gradWh[nodeIndex][offset + k] += gradHidden[nodeIndex] * latterEmb[k];
	        			latterError[k] += gradHidden[nodeIndex] * Wh[nodeIndex][offset+k];
	        		}
	        	}
	        	offset += latterEmb.length;
	        	if(parameters.entityConvolution) {
	        		// continue back-propagate to CNN
	        		entityCNN.backward(latterError, ex.latterIdx, latterState, entityKeeper);
	        	}
	        	
	        	
	        	
	        	
	        }
	        
		}
		
		// Backpropagate gradients on saved pre-computed values to actual embeddings
	    Iterator<Integer> it = toPreCompute.iterator();
	    while(it.hasNext()) {
	    	int x = it.next();
	    	int mapX = preMap.get(x);
	        int tok = x / embFeatureNumber;
	        int offset = (x % embFeatureNumber) * parameters.embeddingSize;
	        for (int j = 0; j < parameters.hiddenSize; ++j) {
	        	
	          for (int k = 0; k < parameters.embeddingSize; ++k) {
	        	keeper.gradWh[j][offset + k] += gradSaved[mapX][j] * owner.getE()[tok][k];
	          }
	          
	          if(tok < owner.getKnownWords1().size() || 
    				  tok >= owner.getKnownWords1().size()+owner.getKnownWords2().size()) {
    			  for (int k = 0; k < parameters.embeddingSize; ++k) 
    				  keeper.gradE[tok][k] += gradSaved[mapX][j] * Wh[j][offset + k];
    		  }  
	          
	        }
	    }
	    
        // L2 Regularization
    	double[][] gradWh = keeper.gradWh;
	    for (int i = 0; i < gradWh.length; ++i) {
	        for (int j = 0; j < gradWh[i].length; ++j) {
	          loss += parameters.regParameter * Wh[i][j] * Wh[i][j] / 2.0;
	          gradWh[i][j] += parameters.regParameter * Wh[i][j];
	        }
	      }
	    
    	double[] gradBh = keeper.gradBh;
    	for (int i = 0; i < gradBh.length; ++i) {
	        loss += parameters.regParameter * Bh[i] * Bh[i] / 2.0;
	        gradBh[i] += parameters.regParameter * Bh[i];
	      }


	      for (int i = 0; i < keeper.gradWo.length; ++i) {
	        for (int j = 0; j < keeper.gradWo[i].length; ++j) {
	          loss += parameters.regParameter * Wo[i][j] * Wo[i][j] / 2.0;
	          keeper.gradWo[i][j] += parameters.regParameter * Wo[i][j];
	        }
	      }
	
	     
    	  for (int i = 0; i < owner.getE().length; ++i) {
    		if(i < owner.getKnownWords1().size() || 
    				  i >= owner.getKnownWords1().size()+owner.getKnownWords2().size()) {
    			for (int j = 0; j < owner.getE()[i].length; ++j) {
  	  	          loss += parameters.regParameter * owner.getE()[i][j] * owner.getE()[i][j] / 2.0;
  	  	          keeper.gradE[i][j] += parameters.regParameter * owner.getE()[i][j];
  	  	        }
        	}
  	        
  	      }
	       
	      
		
		if(debug)
			System.out.println("Cost = " + loss + ", Correct(%) = " + correct);
		
		return keeper;
  
	}
	
	public void updateWeights(GradientKeeper1 keeper) {
		// ada-gradient
			for (int i = 0; i < Wh.length; ++i) {
		      for (int j = 0; j < Wh[i].length; ++j) {
		    	eg2Wh[i][j] += keeper.gradWh[i][j] * keeper.gradWh[i][j];
		        Wh[i][j] -= parameters.adaAlpha * keeper.gradWh[i][j] / Math.sqrt(eg2Wh[i][j] + parameters.adaEps);
		      }
		    }


			for (int i = 0; i < Bh.length; ++i) {
		      eg2Bh[i] += keeper.gradBh[i] * keeper.gradBh[i];
		      Bh[i] -= parameters.adaAlpha * keeper.gradBh[i] / Math.sqrt(eg2Bh[i] + parameters.adaEps);
		    }
	    

	    for (int i = 0; i < Wo.length; ++i) {
	      for (int j = 0; j < Wo[i].length; ++j) {
	        eg2Wo[i][j] += keeper.gradWo[i][j] * keeper.gradWo[i][j];
	        Wo[i][j] -= parameters.adaAlpha * keeper.gradWo[i][j] / Math.sqrt(eg2Wo[i][j] + parameters.adaEps);
	      }
	    }

	    
	    	for (int i = 0; i < owner.getE().length; ++i) {
	    		if(i < owner.getKnownWords1().size() || 
	    				  i >= owner.getKnownWords1().size()+owner.getKnownWords2().size()) {
	    			for (int j = 0; j < owner.getE()[i].length; ++j) {
	    	  	    	owner.getEg2E()[i][j] += keeper.gradE[i][j] * keeper.gradE[i][j];
	    	  	    	owner.getE()[i][j] -= parameters.adaAlpha * keeper.gradE[i][j] / Math.sqrt(owner.getEg2E()[i][j] + parameters.adaEps);
	    	  	      }
	    		}
	  	      
	  	    }
	    
	    
	    if(parameters.entityConvolution)
	    	entityCNN.updateWeights(keeper.entityKeeper);
	    
	    
	}


	/**
	   * Determine the token-position which need to be pre-computed for
	   * training with these examples.
	   */
	public Set<Integer> getToPreCompute(List<Example> examples) {
		Set<Integer> tokPos = new HashSet<Integer>();
	    for (Example ex : examples) {
	    	for(int j=0;j<ex.featureIdx.size();j++) {
	    		int tok = ex.featureIdx.get(j);
	    		if(tok==-1)
	    			continue;
	    		int index = tok*embFeatureNumber+j;
	    		if (preMap.containsKey(index))
	    			tokPos.add(index);
	    	}

	    }
		
	    return tokPos;
	  }
	
	// precompute all tokPos
	public void preCompute() {
	    Set<Integer> keys = preMap.entrySet().stream()
	                              .filter(e -> e.getValue() < parameters.numPreComputed)
	                              .map(Map.Entry::getKey)
	                              .collect(toSet());
	    preCompute(keys);
	  }
	
	// precompute the given tokPos
	public void preCompute(Set<Integer> toPreCompute) {
	
	    saved = new double[preMap.size()][parameters.hiddenSize];
	    
	    Iterator<Integer> it = toPreCompute.iterator();
	    while(it.hasNext()) {
	    	int x = it.next();
		      int mapX = preMap.get(x);
		      int tok = x / embFeatureNumber;
		      int pos = x % embFeatureNumber;
		      for (int j = 0; j < parameters.hiddenSize; ++j)
		        for (int k = 0; k < parameters.embeddingSize; ++k)
		          saved[mapX][j] += Wh[j][pos * parameters.embeddingSize + k]
		          						* owner.getE()[tok][k];
	    }
	    
	    
	    	    
	  }
	
	
}




