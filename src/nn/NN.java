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
import gnu.trove.TIntArrayList;
import gnu.trove.TIntHashSet;
import gnu.trove.TIntIntHashMap;
import gnu.trove.TIntIterator;

public class NN implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1908284566872396134L;
	public Parameters parameters;
	/*
	 *  the weights of the hidden layers
	 *  for first hidden layer, hiddenSize x inputSize
	 *  for other hidden layers, hiddenSize x hiddenSize
	 */
	public List<double[][]> Whs;
	// the bias of the hidden layers, hiddenSize x 1
	public List<double[]> Bhs;
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
	public List<double[][]> eg2Whs; 
	public List<double[]> eg2Bhs;
	public double[][] eg2Wo;
	
	
	public boolean debug;
	
	public Father owner;
	
	public int embFeatureNumber;
	
	public EntityCNN entityCNN;
	public SentenceCNN sentenceCNN;
	
		
	public NN(Parameters parameters, Father owner, TIntArrayList preComputed, Example example) {
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
		
		
		Whs = new ArrayList<>();
		eg2Whs = new ArrayList<>();
		for(int k=0;k<parameters.hiddenLevel;k++) {
			if(k==0) {
				double[][] Wh = new double[parameters.hiddenSize][inputSize];
				double[][] eg2Wh = new double[Wh.length][Wh[0].length];
				Whs.add(Wh);
				eg2Whs.add(eg2Wh);
				for(int i=0;i<Wh.length;i++) {
					for(int j=0;j<Wh[0].length;j++) {
						Wh[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
					}
				}
			} else {
				double[][] Wh = new double[parameters.hiddenSize][parameters.hiddenSize];
				double[][] eg2Wh = new double[Wh.length][Wh[0].length];
				Whs.add(Wh);
				eg2Whs.add(eg2Wh);
				for(int i=0;i<Wh.length;i++) {
					for(int j=0;j<Wh[0].length;j++) {
						Wh[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
					}
				}
			}
		}
		
		
		Bhs = new ArrayList<>();
		eg2Bhs = new ArrayList<>();
		for(int k=0;k<parameters.hiddenLevel;k++) {
			double[] Bh = new double[parameters.hiddenSize];
			double[] eg2Bh = new double[Bh.length];
			Bhs.add(Bh);
			eg2Bhs.add(eg2Bh);
			for(int i=0;i<Bh.length;i++) {
				Bh[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		
		
		preMap = new HashMap<>();
	    for (int i = 0; i < preComputed.size(); ++i)
	      preMap.put(preComputed.get(i), i);
	    
	    if(parameters.entityConvolution)
	    	entityCNN = new EntityCNN(parameters, owner, debug);
	    
	    if(parameters.sentenceConvolution) 
	    	sentenceCNN = new SentenceCNN(parameters, owner, debug);
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
			if(parameters.entityPooling == 1) {
				double[] emb = new double[parameters.entityDimension];
				for(int i=0;i<entityIdx.size();i++) {
					int embIdx = entityIdx.get(i);
					for(int j=0;j<parameters.entityDimension;j++) {
						emb[j] += owner.getE()[embIdx][j]/entityIdx.size();
					}
				}
				State state = new State();
				state.emb = emb;
				return state;
																
			} else if(parameters.entityPooling == 2) {
				double[] emb = Arrays.copyOf(owner.getE()[entityIdx.get(0)], parameters.entityDimension);
				for(int j=0;j<parameters.entityDimension;j++) {
					for(int i=1;i<entityIdx.size();i++) {
						if(emb[j] < owner.getE()[entityIdx.get(i)][j])
							emb[j] = owner.getE()[entityIdx.get(i)][j];
					}
					
				}
				State state = new State();
				state.emb = emb;
				return state;
				
			} else {
				throw new Exception();
			}
		}
	}
	
	/*
	 * Given a example, compute from bottom to up and get the output scores.
	 * This cannot be used in training, because we don't consider dropout.
	 */
	public double[] computeScores(Example ex) throws Exception {
		
		double[] scores = new double[parameters.outputSize];
			
		double[] lastHidden = null;
		double[] lastHidden3 = null;
		for(int level = 0; level<Whs.size();level++) {
				        
	        double[] hidden = new double[parameters.hiddenSize];
	        if(level==0) { // input -> hidden1
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
		                hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset + k] * owner.getE()[tok][k];
		            }
		          }
		          
		        }
		        
		        if(ex.bRelation) {
		        	double[] formerEmb = generateEntityComposite(ex.formerIdx).emb;
		        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
		        		for (int k = 0; k < formerEmb.length; ++k)
		        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset+k] * formerEmb[k];
		        	}
		        	offset += formerEmb.length;
		        	
		        	double[] latterEmb = generateEntityComposite(ex.latterIdx).emb;
		        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
		        		for (int k = 0; k < latterEmb.length; ++k)
		        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset+k] * latterEmb[k];
		        	}
		        	offset += latterEmb.length;
		        	
		        	if(parameters.sentenceConvolution) {
			        	double[] sentenceEmb = sentenceCNN.forward(ex).emb;
			        	for(int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
			        		for(int k=0; k< sentenceEmb.length; ++k) {
			        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset+k] * sentenceEmb[k];
			        		}
			        	}
			        	offset += sentenceEmb.length;
		        	}
		        	
		        }
	        } else { // hidden -> hidden
	        	
	        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
	        		for(int k = 0; k < parameters.hiddenSize; k++) {
	        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][k] * lastHidden[k];
	        		}
	        	}
	        	
	        }
	        
	        // add bias and activation
	        double[] hidden3 = new double[parameters.hiddenSize];
	        for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++) {
	            hidden[nodeIndex] += Bhs.get(level)[nodeIndex];
	            if(parameters.actFuncOfHidden == 1) {
	            	hidden3[nodeIndex] = Math.pow(hidden[nodeIndex], 3);
				} else {
					throw new Exception();
				}
	            
	        }
	        
	        lastHidden = hidden;
	        lastHidden3 = hidden3;
		}
		
		
        // hidden -> output
		for (int i = 0; i < parameters.outputSize; ++i) {
        	for (int nodeIndex=0; nodeIndex<parameters.hiddenSize; nodeIndex++)
              scores[i] += Wo[i][nodeIndex] * lastHidden3[nodeIndex];

        }
		
		
        return scores;
	}
		
	/*
	 * Given some examples, compute from bottom to top for each example
	 * back-propagate their gradient
	 */
	public GradientKeeper process(List<Example> examples, Perceptron perceptron) throws Exception {
		// precompute
		Set<Integer> toPreCompute = getToPreCompute(examples);
	    preCompute(toPreCompute);
	    
	    gradSaved = new double[preMap.size()][parameters.hiddenSize];
	    
	    GradientKeeper keeper = new GradientKeeper(parameters, this);
	    EntityCNNGradientKeeper entityKeeper = null;
	    if(parameters.entityConvolution)
	    	entityKeeper = new EntityCNNGradientKeeper(parameters, entityCNN, keeper);
	    keeper.entityKeeper = entityKeeper;
	    SentenceCNNGradientKeeper sentenceKeeper = null;
	    if(parameters.sentenceConvolution) 
	    	sentenceKeeper = new SentenceCNNGradientKeeper(parameters, sentenceCNN, keeper);
	    keeper.sentenceKeeper = sentenceKeeper;
		
	    double loss = 0;
		double correct = 0;
		Random random = new Random(System.currentTimeMillis());
		
		// mini-batch
		for (Example ex : examples) {
			
			List<double[]> hiddens = new ArrayList<>();
			List<double[]> hidden3s = new ArrayList<>();
			double[] scores = new double[parameters.outputSize];
			
			List<int []> lses = new ArrayList<>();
			for(int level = 0; level<Whs.size();level++) {
				// Run dropout: randomly drop some hidden-layer units. `ls`
		        // contains the indices of those units which are still active
				int [] ls = IntStream.range(0, parameters.hiddenSize)
		                            .filter(n-> random.nextDouble() > parameters.dropProb)
		                            .toArray();
				lses.add(ls);
		        
		        double[] hidden = new double[parameters.hiddenSize];
		        hiddens.add(hidden);
		        if(level==0) { // input -> hidden1
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
			                hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset + k] * owner.getE()[tok][k];
			            }
			          }
			          
			        }
			        
			        if(ex.bRelation) {
			        	double[] formerEmb = generateEntityComposite(ex.formerIdx).emb;
			        	for(int nodeIndex : ls) {
			        		for (int k = 0; k < formerEmb.length; ++k)
			        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset+k] * formerEmb[k];
			        	}
			        	offset += formerEmb.length;
			        	
			        	double[] latterEmb = generateEntityComposite(ex.latterIdx).emb;
			        	for(int nodeIndex : ls) {
			        		for (int k = 0; k < latterEmb.length; ++k)
			        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset+k] * latterEmb[k];
			        	}
			        	offset += latterEmb.length;
			        	
			        	if(parameters.sentenceConvolution) {
				        	double[] sentenceEmb = sentenceCNN.forward(ex).emb;
				        	for(int nodeIndex : ls) {
				        		for (int k = 0; k < sentenceEmb.length; ++k) {
				        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][offset+k] * sentenceEmb[k];
				        		}
				        	}
				        	offset += sentenceEmb.length;
			        	}
			        }
		        } else { // hidden -> hidden
		        	double[] lastHidden = hiddens.get(level-1);
		        	for(int nodeIndex : ls) {
		        		for(int k = 0; k < parameters.hiddenSize; k++) {
		        			hidden[nodeIndex] += Whs.get(level)[nodeIndex][k] * lastHidden[k];
		        		}
		        	}
		        	
		        }
		        
		        // add bias and activation
		        double[] hidden3 = new double[parameters.hiddenSize];
		        hidden3s.add(hidden3);
		        for (int nodeIndex : ls) {
		            hidden[nodeIndex] += Bhs.get(level)[nodeIndex];
		            if(parameters.actFuncOfHidden == 1) {
		            	hidden3[nodeIndex] = Math.pow(hidden[nodeIndex], 3);
					} else {
						throw new Exception();
					}
		            
		        }
			}
			
			
	        // hidden -> output
			int optLabel = -1; // the label with the highest score
	        for (int i = 0; i < parameters.outputSize; ++i) {
	            for (int nodeIndex : lses.get(lses.size()-1))
	              scores[i] += Wo[i][nodeIndex] * hidden3s.get(parameters.hiddenLevel-1)[nodeIndex];

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
	        
	        // perceptron
	        if(owner instanceof PerceptronNNADE) {
	        	HashSet<String> features = null;
	        	if(ex.bRelation) {
	        		features = perceptron.featureFunction(ex.tokens, -1, PerceptronNNADE.transitionNumber2String(optLabel), ex.drug, ex.disease);
	        	} else {
	        		features = perceptron.featureFunction(ex.tokens, ex.idx, PerceptronNNADE.transitionNumber2String(optLabel), null, null); 
	        	}
	        	SparseVector preVector = perceptron.featuresToSparseVector(features);
        		SparseVector goldVector = perceptron.featuresToSparseVector(ex.goldFeatures);
        		SparseVector temp = goldVector.vectorAdd(preVector, -1);
				perceptron.w = perceptron.w.vectorAdd(temp, 1);
	        	
	        	/*features.retainAll(ex.goldFeatures);
	        	for(String gold:ex.goldFeatures) {
	        		if(!features.contains(gold)) {
	        			perceptron.w[perceptron.getFeatureIndex(gold)] += 1;
	        		}
	        	}*/

	        }
	        
			// output -> highest hidden layer
	        double[] gradHidden3 = new double[parameters.hiddenSize];
	        for (int i = 0; i < parameters.outputSize; ++i) {
	          
	            double temp = -(ex.label[i] - scores[i] / sum2) / examples.size();
	            for (int nodeIndex : lses.get(lses.size()-1)) {
	              keeper.gradWo[i][nodeIndex] += temp * hidden3s.get(hidden3s.size()-1)[nodeIndex];
	              gradHidden3[nodeIndex] += temp * Wo[i][nodeIndex];
	            }
	        }
	        
	        // highest hidden layer -> hidden layer or input layer
	        for(int level=parameters.hiddenLevel-1;level>=0;level--) {
	        	int [] lsCurrent = lses.get(level); // activated nodes for the current hidden layer
	        	int [] lsLower = null; // activated nodes for the next lower hidden layer
	        	if(level > 0)
	        		lsLower = lses.get(level-1); 

	        	double[] gradHidden = new double[parameters.hiddenSize];
		        for (int nodeIndex : lsCurrent) {
		        	if(parameters.actFuncOfHidden == 1)
		        		gradHidden[nodeIndex] = gradHidden3[nodeIndex] * 3 * hiddens.get(level)[nodeIndex] * hiddens.get(level)[nodeIndex];
		        	else
		        		throw new Exception();
		        	keeper.gradBhs.get(level)[nodeIndex] += gradHidden[nodeIndex]; // in chen 2014 EMNLP, this is gradHidden3 but I think it's wrong
		        }
		        
		        if(level==0) {
		        	int offset = 0;
			        for (int j = 0; j < embFeatureNumber; ++j, offset += parameters.embeddingSize) {
			          int tok = ex.featureIdx.get(j);
			          if(tok==-1)
			        	  continue;
			          int index = tok * embFeatureNumber + j;
			          if (preMap.containsKey(index)) {
			            int id = preMap.get(index);
			            for (int nodeIndex : lsCurrent)
			              gradSaved[id][nodeIndex] += gradHidden[nodeIndex];
			          } else {
			            for (int nodeIndex : lsCurrent) {
			              for (int k = 0; k < parameters.embeddingSize; ++k) {
			            	keeper.gradWhs.get(level)[nodeIndex][offset + k] += gradHidden[nodeIndex] * owner.getE()[tok][k];
			                if(parameters.bEmbeddingFineTune)
			                	keeper.gradE[tok][k] += gradHidden[nodeIndex] * Whs.get(level)[nodeIndex][offset + k];
			              }
			            }
			          }
			          
			        }
			        
			        if(ex.bRelation) {
			        	
			        	State formerState = generateEntityComposite(ex.formerIdx);
			        	double[] formerEmb = formerState.emb;
			        	double[] formerError = new double[formerEmb.length];
			        	for(int nodeIndex : lsCurrent) {
			        		for (int k = 0; k < formerEmb.length; ++k) {
			        			keeper.gradWhs.get(level)[nodeIndex][offset + k] += gradHidden[nodeIndex] * formerEmb[k];
			        			formerError[k] += gradHidden[nodeIndex] * Whs.get(level)[nodeIndex][offset+k];
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
			        	for(int nodeIndex : lsCurrent) {
			        		for (int k = 0; k < latterEmb.length; ++k) {
			        			keeper.gradWhs.get(level)[nodeIndex][offset + k] += gradHidden[nodeIndex] * latterEmb[k];
			        			latterError[k] += gradHidden[nodeIndex] * Whs.get(level)[nodeIndex][offset+k];
			        		}
			        	}
			        	offset += latterEmb.length;
			        	if(parameters.entityConvolution) {
			        		// continue back-propagate to CNN
			        		entityCNN.backward(latterError, ex.latterIdx, latterState, entityKeeper);
			        	}
			        	
			        	if(parameters.sentenceConvolution) {
			        		State sentenceState = sentenceCNN.forward(ex);
			        		double[] sentenceEmb = sentenceState.emb;
			        		double[] sentenceError = new double[sentenceEmb.length];
			        		for(int nodeIndex : lsCurrent) {
			        			for(int k=0; k<sentenceEmb.length; ++k) {
			        				keeper.gradWhs.get(level)[nodeIndex][offset + k] += gradHidden[nodeIndex] * sentenceEmb[k];
			        				sentenceError[k] += gradHidden[nodeIndex] * Whs.get(level)[nodeIndex][offset+k];
			        			}
			        		}
			        		offset += sentenceEmb.length;
			        		sentenceCNN.backward(sentenceError, ex, sentenceState, sentenceKeeper);
			        	}
			        	
			        	
			        }
			        
		        } else {
		        	// only consider the activated hidden nodes in both hidden layers
		        	for (int nodeIndex : lsCurrent) {
		        		for(int lowerIndex : lsLower) {
		        			keeper.gradWhs.get(level)[nodeIndex][lowerIndex] += gradHidden[nodeIndex] * hidden3s.get(level-1)[lowerIndex];
		        		}
		        	}
		        	
		        }
		        
		        
	        	// recompute gradHidden3 for the next layer, except the lowest hidden layer
		        if(level != 0) {
	        		gradHidden3 = new double[parameters.hiddenSize];
	        		for (int nodeIndex : lsLower) {
	        			for(int j=0; j<parameters.hiddenSize; j++)
	        				gradHidden3[nodeIndex] += gradHidden[j] * Whs.get(level)[j][nodeIndex];
	        		}
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
	        	keeper.gradWhs.get(0)[j][offset + k] += gradSaved[mapX][j] * owner.getE()[tok][k];
	            if(parameters.bEmbeddingFineTune)
	            	keeper.gradE[tok][k] += gradSaved[mapX][j] * Whs.get(0)[j][offset + k];
	          }
	        }
	    }
	    
        // L2 Regularization
	    for(int k=0;k<keeper.gradWhs.size();k++) {
	    	double[][] gradWh = keeper.gradWhs.get(k);
		    for (int i = 0; i < gradWh.length; ++i) {
		        for (int j = 0; j < gradWh[i].length; ++j) {
		          loss += parameters.regParameter * Whs.get(k)[i][j] * Whs.get(k)[i][j] / 2.0;
		          gradWh[i][j] += parameters.regParameter * Whs.get(k)[i][j];
		        }
		      }
	    }
	    
	    for(int k=0;k<keeper.gradBhs.size();k++) {
	    	double[] gradBh = keeper.gradBhs.get(k);
	    	for (int i = 0; i < gradBh.length; ++i) {
		        loss += parameters.regParameter * Bhs.get(k)[i] * Bhs.get(k)[i] / 2.0;
		        gradBh[i] += parameters.regParameter * Bhs.get(k)[i];
		      }
	    }
	
	      
	
	      for (int i = 0; i < keeper.gradWo.length; ++i) {
	        for (int j = 0; j < keeper.gradWo[i].length; ++j) {
	          loss += parameters.regParameter * Wo[i][j] * Wo[i][j] / 2.0;
	          keeper.gradWo[i][j] += parameters.regParameter * Wo[i][j];
	        }
	      }
	
	      if(parameters.bEmbeddingFineTune) {
	    	  for (int i = 0; i < owner.getE().length; ++i) {
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
	
	public void updateWeights(GradientKeeper keeper) {
		// ada-gradient
		for(int k=0;k<Whs.size();k++) {
			double[][] Wh = Whs.get(k);
			for (int i = 0; i < Wh.length; ++i) {
		      for (int j = 0; j < Wh[i].length; ++j) {
		    	eg2Whs.get(k)[i][j] += keeper.gradWhs.get(k)[i][j] * keeper.gradWhs.get(k)[i][j];
		        Wh[i][j] -= parameters.adaAlpha * keeper.gradWhs.get(k)[i][j] / Math.sqrt(eg2Whs.get(k)[i][j] + parameters.adaEps);
		      }
		    }
		}

		for(int k=0;k<Bhs.size();k++) {
			double[] Bh = Bhs.get(k);
			for (int i = 0; i < Bh.length; ++i) {
		      eg2Bhs.get(k)[i] += keeper.gradBhs.get(k)[i] * keeper.gradBhs.get(k)[i];
		      Bh[i] -= parameters.adaAlpha * keeper.gradBhs.get(k)[i] / Math.sqrt(eg2Bhs.get(k)[i] + parameters.adaEps);
		    }
		}
	    

	    for (int i = 0; i < Wo.length; ++i) {
	      for (int j = 0; j < Wo[i].length; ++j) {
	        eg2Wo[i][j] += keeper.gradWo[i][j] * keeper.gradWo[i][j];
	        Wo[i][j] -= parameters.adaAlpha * keeper.gradWo[i][j] / Math.sqrt(eg2Wo[i][j] + parameters.adaEps);
	      }
	    }

	    if(parameters.bEmbeddingFineTune) {
	    	for (int i = 0; i < owner.getE().length; ++i) {
	  	      for (int j = 0; j < owner.getE()[i].length; ++j) {
	  	    	owner.getEg2E()[i][j] += keeper.gradE[i][j] * keeper.gradE[i][j];
	  	    	owner.getE()[i][j] -= parameters.adaAlpha * keeper.gradE[i][j] / Math.sqrt(owner.getEg2E()[i][j] + parameters.adaEps);
	  	      }
	  	    }
	    }
	    
	    if(parameters.entityConvolution)
	    	entityCNN.updateWeights(keeper.entityKeeper);
	    
	    if(parameters.sentenceConvolution)
	    	sentenceCNN.updateWeights(keeper.sentenceKeeper);
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
		          saved[mapX][j] += Whs.get(0)[j][pos * parameters.embeddingSize + k]
		          						* owner.getE()[tok][k];
	    }
	    
	    
	    	    
	  }
	
	/*
	 *  Given an example, compute its loss.
	 *  This is only used by gradient checking.
	 */
	public double computeLoss(Example ex) throws Exception {
		
		double[] scores = computeScores(ex);
        
		int optLabel = -1; // the label with the highest score
        for (int i = 0; i < parameters.outputSize; ++i) {
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

        double loss = (Math.log(sum2) - Math.log(sum1));
        
        return loss;
	}
	
	public void checkGradients (GradientKeeper keeper, List<Example> examples) throws Exception {
		// we only check Whs, Wbs and Wo
		System.out.println(Parameters.SEPARATOR+" Gradient checking begin");
		
		// randomly select one point
		Random random = new Random(System.currentTimeMillis());
		
		// Wo
		{
			int check_i = random.nextInt(Wo.length);
			int check_j = random.nextInt(Wo[0].length);
			double orginValue = Wo[check_i][check_j];
			
		  Wo[check_i][check_j] = orginValue + parameters.epsilonGradientCheck;
		  double lossAdd = 0.0;
		  // because Wo was computed based on all examples, we should compute mockGrad following this way
		  for (int i = 0; i < examples.size(); i++) {
		    Example oneExam = examples.get(i);
		    lossAdd += computeLoss(oneExam);
		  }
	
		  Wo[check_i][check_j] = orginValue - parameters.epsilonGradientCheck;
		  double lossSub = 0.0;
		  for (int i = 0; i < examples.size(); i++) {
		    Example oneExam = examples.get(i);
		    lossSub += computeLoss(oneExam);
		  }
	
		  double mockGrad = (lossAdd - lossSub) / (parameters.epsilonGradientCheck * 2);
		  mockGrad = mockGrad / examples.size();
		  double computeGrad = keeper.gradWo[check_i][check_j];
	
		  System.out.printf("Wo[%d][%d] abs(mockGrad-computeGrad)= %.18f\n", check_i, check_j, Math.abs(mockGrad-computeGrad));
		  
		  // restore the value, important!!!
		  Wo[check_i][check_j] =  orginValue;
		}
		
		// Whs
		{
			for(int level=0;level<Whs.size();level++) {
				double[][] Wh = Whs.get(level);
				int check_i = random.nextInt(Wh.length);
				int check_j = random.nextInt(Wh[0].length);
				double orginValue = Wh[check_i][check_j];
				
				Wh[check_i][check_j] = orginValue + parameters.epsilonGradientCheck;
				double lossAdd = 0.0;
			  for (int i = 0; i < examples.size(); i++) {
			    Example oneExam = examples.get(i);
			    lossAdd += computeLoss(oneExam);
			  }
		
			  Wh[check_i][check_j] = orginValue - parameters.epsilonGradientCheck;
			  double lossSub = 0.0;
			  for (int i = 0; i < examples.size(); i++) {
			    Example oneExam = examples.get(i);
			    lossSub += computeLoss(oneExam);
			  }
		
			  double mockGrad = (lossAdd - lossSub) / (parameters.epsilonGradientCheck * 2);
			  mockGrad = mockGrad / examples.size();
			  double computeGrad = keeper.gradWhs.get(level)[check_i][check_j];
		
			  System.out.printf("Wh%d[%d][%d] abs(mockGrad-computeGrad)= %.18f\n", level+1, check_i, check_j, Math.abs(mockGrad-computeGrad));
			  
			  // restore the value, important!!!
			  Wh[check_i][check_j] =  orginValue;
			}
			
		}
		
		// Bhs
		{
			for(int level=0;level<Bhs.size();level++) {
				double[] Bh = Bhs.get(level);
				int check_i = random.nextInt(Bh.length);
				double orginValue = Bh[check_i];
				
				Bh[check_i] = orginValue + parameters.epsilonGradientCheck;
				double lossAdd = 0.0;
			  for (int i = 0; i < examples.size(); i++) {
			    Example oneExam = examples.get(i);
			    lossAdd += computeLoss(oneExam);
			  }
		
			  Bh[check_i] = orginValue - parameters.epsilonGradientCheck;
			  double lossSub = 0.0;
			  for (int i = 0; i < examples.size(); i++) {
			    Example oneExam = examples.get(i);
			    lossSub += computeLoss(oneExam);
			  }
		
			  double mockGrad = (lossAdd - lossSub) / (parameters.epsilonGradientCheck * 2);
			  mockGrad = mockGrad / examples.size();
			  double computeGrad = keeper.gradBhs.get(level)[check_i];
		
			  System.out.printf("Bh%d[%d] abs(mockGrad-computeGrad)= %.18f\n", level+1, check_i, Math.abs(mockGrad-computeGrad));
			  
			  // restore the value, important!!!
			  Bh[check_i] =  orginValue;
			}
			
		}
		
		
		
		System.out.println(Parameters.SEPARATOR+" Gradient checking end");
	}
	
	
	
}

class State {
	double[] emb;
	int[] maxRemember;
	double[][] S;
}

class GradientKeeper {
	public List<double[][]> gradWhs;
	public List<double[]> gradBhs;
	public double[][] gradWo;
	public double[][] gradE;
	public EntityCNNGradientKeeper entityKeeper;
	public SentenceCNNGradientKeeper sentenceKeeper;
	
		
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public GradientKeeper(Parameters parameters, NN nn) {
		gradWhs = new ArrayList<>();
	    for(int k=0;k<parameters.hiddenLevel;k++) {
	    	double[][] gradWh = new double[nn.Whs.get(k).length][nn.Whs.get(k)[0].length];
			gradWhs.add(gradWh);
		}
		gradBhs = new ArrayList<>();
		for(int k=0;k<parameters.hiddenLevel;k++) {
			double[] gradBh = new double[nn.Bhs.get(k).length];
			gradBhs.add(gradBh);
		}
		
		gradWo = new double[nn.Wo.length][nn.Wo[0].length];
		gradE = null;
		if(parameters.bEmbeddingFineTune)	
			gradE = new double[nn.owner.getE().length][nn.owner.getE()[0].length];
			
	}
}

class GradientKeeper1 {
	public double[][] gradWh;
	public double[] gradBh;
	public double[][] gradWo;
	public double[][] gradE;
	public EntityCNNGradientKeeper entityKeeper;
	public SentenceCNNGradientKeeper sentenceKeeper;
	
		
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public GradientKeeper1(Parameters parameters, NNSimple nn) {
		gradWh = new double[nn.Wh.length][nn.Wh[0].length];
		
		gradBh = new double[nn.Bh.length];
		
		
		gradWo = new double[nn.Wo.length][nn.Wo[0].length];
		gradE = null;
		if(parameters.bEmbeddingFineTune)	
			gradE = new double[nn.owner.getE().length][nn.owner.getE()[0].length];
			
	}
}
