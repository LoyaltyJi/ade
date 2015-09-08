package pipeline;

import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;
import gnu.trove.TObjectDoubleHashMap;
import gnu.trove.TObjectIntHashMap;

import java.io.Serializable;
import java.util.ArrayList;

import joint.PerceptronInputData1;
import cc.mallet.types.SparseVector;
import cn.fox.machine_learning.Perceptron;
import cn.fox.machine_learning.PerceptronFeatureFunction;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import drug_side_effect_utils.Entity;



public class PerceptronEntity extends Perceptron {

	private static final long serialVersionUID = -589040720903919365L;
	// the maximum length of each alphabet1 type
	private TIntArrayList d;
	
	public PerceptronEntity(ArrayList<String> alphabet1, TIntArrayList d, float convergeThreshold, double weightMax) {
		super(alphabet1, null, convergeThreshold, weightMax);
		
		this.alphabet1.add(0, Perceptron.EMPTY);		
				
		this.d = new TIntArrayList();
		this.d.add(1);
		for(int i=0;i<d.size();i++) {
			this.d.add(d.get(i));
		}
		
		
	}
	
	@Override
	public void buildFeatureAlphabet(ArrayList<PerceptronInputData> inputDatas, ArrayList<PerceptronOutputData> outputDatas, Object other) {
		try {
			w1 = new SparseVector();
			
			
			// use gold data and feature functions to build alphabet and gold feature vectors
			for(int i=0;i<inputDatas.size();i++)  {
				for(int j=0;j<inputDatas.get(i).tokens.size();j++) { // we add feature vector according to the token index
					PerceptronStatus status = new PerceptronStatus(null, j, 0); // we define this step is 0.
					f(inputDatas.get(i), status, outputDatas.get(i), other);
				}
			}

			
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public PerceptronStatus beamSearch(PerceptronInputData x,
			PerceptronOutputData y, boolean isTrain, int beamSize, Object other)
			throws Exception {
		// each token in the input data corresponds a beam 
		ArrayList<ArrayList<PerceptronOutputData>> beams = new ArrayList<ArrayList<PerceptronOutputData>>();
		for(int i=0;i<x.tokens.size();i++)
			beams.add(new ArrayList<PerceptronOutputData>());
		
		
		for(int i=0;i<x.tokens.size();i++) {
			ArrayList<PerceptronOutputData> buf = new ArrayList<PerceptronOutputData>();
			for(int t=0;t<this.alphabet1.size();t++) {
				if(i==0) {
					buf.add(PerceptronEntityOutputData.append(null, alphabet1.get(t), x, 0, 0));
					continue;
				}
				for(int dd=1;dd<=this.d.get(t);dd++) {
					if(i-dd>=0) {
						for(int yy=0;yy<beams.get(i-dd).size();yy++) {
							int k = i-dd+1;
							buf.add(PerceptronEntityOutputData.append(beams.get(i-dd).get(yy), alphabet1.get(t), x, k, i));
						}
					} else if(i-dd==-1){ 
						buf.add(PerceptronEntityOutputData.append(null, alphabet1.get(t), x, 0, i));
						break;
					}
				}
			}
			
			PerceptronStatus statusKBest = new PerceptronStatus(null, i, 1);
			kBest(x, statusKBest, beams.get(i), buf, beamSize, other);
			// early update
			if(isTrain) {
				int m=0;
				for(;m<beams.get(i).size();m++) {
					if(beams.get(i).get(m).isIdenticalWith(x, y, statusKBest)) {
						break;
					}
				}
				if(m==beams.get(i).size() && isAlignedWithGold(beams.get(i).get(0), y, i)) {
					PerceptronStatus returnType = new PerceptronStatus(beams.get(i).get(0), i, 1);
					return returnType;
				}
			}

			
		}
		
		PerceptronStatus returnType = new PerceptronStatus(beams.get(x.tokens.size()-1).get(0), x.tokens.size()-1, 3);
		return returnType;
	}

	
	public static boolean isAlignedWithGold(PerceptronOutputData predict, PerceptronOutputData gold, int tokenIndex) {
		PerceptronEntityOutputData predict1 = (PerceptronEntityOutputData)predict;
		PerceptronEntityOutputData gold1 = (PerceptronEntityOutputData)gold;
		
		Entity predictLastSeg = predict1.getLastSegment(tokenIndex);
		Entity goldLastSeg = gold1.getLastSegment(tokenIndex);
		if(predictLastSeg.end==goldLastSeg.end)
			return true;
		else return false;
	}
	

	@Override	
	public void setFeatureFunction(ArrayList<PerceptronFeatureFunction> featureFunctions1, ArrayList<PerceptronFeatureFunction> featureFunctions2) {
		this.featureFunctions1 = featureFunctions1;
	}
	
	@Override	
	public void normalizeWeight() {
		// norm
		double norm1 = w1.twoNorm();
		for(int j=0;j<w1.getIndices().length;j++) {
			w1.setValueAtLocation(j, w1.valueAtLocation(j)/norm1);
		}
	
	}

	@Override
	public void trainPerceptron(int T, int beamSize, ArrayList<PerceptronInputData> input, ArrayList<PerceptronOutputData> output, Object other) {
		try {		
			for(int i=0;i<T;i++) {
				SparseVector old1 = (SparseVector)w1.cloneMatrix();
				
				long startTime = System.currentTimeMillis();
				//Matrix w_copy = w.copy();
				for(int j=0;j<input.size();j++) {
					PerceptronInputData x = input.get(j);
					PerceptronOutputData y = output.get(j);
					// get the best predicted answer
					PerceptronStatus status = beamSearch(x, y, true, beamSize, other);
					
					if(!status.z.isIdenticalWith(x, y, status)) {
						// if the predicted answer are not identical to the gold answer, update the model.

						if(status.step==1) {
							SparseVector fxy  = f(x, status, y, other).sv1;
							SparseVector fxz = f(x, status, status.z, other).sv1;
							SparseVector temp = fxy.vectorAdd(fxz, -1);
							w1 = w1.vectorAdd(temp, 1);
						}
						else if(status.step==3) {
							FReturnType rtFxy = f(x, status, y, other);
							FReturnType rtFxz = f(x, status, status.z, other);
							
							SparseVector fxy1  = rtFxy.sv1;
							SparseVector fxz1 = rtFxz.sv1;
							SparseVector temp1 = fxy1.vectorAdd(fxz1, -1);
							w1 = w1.vectorAdd(temp1, 1);
							

						} else
							throw new Exception();
						
						
					}
				}
				// check weight
				if(weightMax!=0) {
					double values1[] = w1.getValues();
					for(int j=0;j<values1.length;j++) {
						if(values1[j]>weightMax)
							values1[j] = weightMax;
						else if(values1[j]<-weightMax)
							values1[j] = -weightMax;
					}
					
				}
			
				
				float dist1 = (float)old1.vectorAdd(w1, -1).twoNorm();
				float dist = dist1;
				
				
				if(dist<convergeThreshold) {
					System.out.println("converged, quit training");
					normalizeWeight();
					return;
				} 
				else {
					long endTime = System.currentTimeMillis();
					System.out.println((i+1)+" train finished "+(dist1)+" "+(endTime-startTime)+" ms");
				}

			}
			System.out.println("achieve max training times, quit");
		} catch(Exception e) {
			e.printStackTrace();
		}
		// norm
		normalizeWeight();
		return;
	}
	
	
	@Override
	public void kBest(PerceptronInputData x, PerceptronStatus status, ArrayList<PerceptronOutputData> beam, ArrayList<PerceptronOutputData> buf, int beamSize, Object other)throws Exception {
		// compute all the scores in the buf
		TDoubleArrayList scores = new TDoubleArrayList();
		for(PerceptronOutputData y:buf) {
			FReturnType ret = f(x,status,y, other);
			if(status.step==1) {
				scores.add(w1.dotProduct(ret.sv1));
			}else 
				throw new Exception();
			
		}
		
		// assign k best to the beam, and note that buf may be more or less than beamSize.
		int K = buf.size()>beamSize ? beamSize:buf.size();
		PerceptronOutputData[] temp = new PerceptronOutputData[K];
		Double[] tempScore = new Double[K];
		for(int i=0;i<buf.size();i++) {
			for(int j=0;j<K;j++) {
				if(temp[j]==null || scores.get(i)>tempScore[j]) {
					if(temp[j] != null) {
						for(int m=K-2;m>=j;m--) {
							temp[m+1] = temp[m];
							tempScore[m+1] = tempScore[m];
						}
					}
					
					temp[j] = buf.get(i);
					tempScore[j] = scores.get(i);
					break;
				}
			}
		}
		
		beam.clear();
		for(int i=0;i<K;i++) {
			beam.add(temp[i]);
		}
		
		return;
	}
	
	
	// Compute the feature vector "f" based on the current status.
	protected FReturnType f(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other) throws Exception {	
		
		if(y.isGold) {
			if(status.step==0) { // initialize the feature vectors of gold output
				TObjectDoubleHashMap<String> map1 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions1.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions1.get(j);
					featureFunction.compute(x, status, y, other, map1);
				}
				y.featureVectors1.add(hashMapToSparseVector(map1));
				
				
				return new FReturnType(y.featureVectors1.get(status.tokenIndex), null);
			} else if(status.step==1) {
				return new FReturnType(y.featureVectors1.get(status.tokenIndex), null);
			} else if(status.step==3) {
				return new FReturnType(y.featureVectors1.get(status.tokenIndex), null);
			} else
				throw new Exception();
			
		} else {
			if(status.step==1) {
				TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions1.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions1.get(j);
					featureFunction.compute(x, status, y, other, map);
				}
				y.featureVector1 = hashMapToSparseVector(map);
				return new FReturnType(y.featureVector1, null);
			} else if(status.step==3) {
				TObjectDoubleHashMap<String> map1 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions1.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions1.get(j);
					featureFunction.compute(x, status, y, other, map1);
				}
				y.featureVector1 = hashMapToSparseVector(map1);
				
				
				return new FReturnType(y.featureVector1, null);
			} else
				throw new Exception();
			
		}

	}
	
		
	


}
