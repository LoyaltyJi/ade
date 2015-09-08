package pipeline;

import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;
import gnu.trove.TObjectDoubleHashMap;
import gnu.trove.TObjectIntHashMap;

import java.io.Serializable;
import java.util.ArrayList;

import joint.PerceptronOutputData1;
import cc.mallet.types.SparseVector;
import cn.fox.machine_learning.Perceptron;
import cn.fox.machine_learning.PerceptronFeatureFunction;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;

public class PerceptronRelation extends Perceptron {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7077106935494560143L;

	@Override
	public void buildFeatureAlphabet(ArrayList<PerceptronInputData> inputDatas, ArrayList<PerceptronOutputData> outputDatas, Object other) {
		try {
			w2 = new SparseVector();
			
			// use gold data and feature functions to build alphabet and gold feature vectors
			for(int i=0;i<inputDatas.size();i++)  {
				for(int j=0;j<inputDatas.get(i).tokens.size();j++) { // we add feature vector according to the token index
					PerceptronStatus status = new PerceptronStatus(null, j, 0); // we define this step is 0.
					FReturnType ret = f(inputDatas.get(i), status, outputDatas.get(i), other);
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
			
			// put gold entities into beam
			PerceptronOutputData1 y1 = (PerceptronOutputData1)y;
			int segmentIndex = y1.getLastSegmentIndex(i);
			Entity lastSegment = y1.segments.get(segmentIndex);
			if(lastSegment.end==i) {
				// there is a entity ended at i
				
				if(i==0) {
					PerceptronOutputData1 temp = (PerceptronOutputData1) PerceptronOutputData1.append(null, lastSegment.type, x, lastSegment.start, lastSegment.end);;
					beams.get(i).add(temp);
				} else {
					for(int k=0;k<beams.get(i-1).size();k++) {
						PerceptronOutputData1 temp = (PerceptronOutputData1)beams.get(i-1).get(k);
						temp = (PerceptronOutputData1) PerceptronOutputData1.append(temp, lastSegment.type, x, lastSegment.start, lastSegment.end);
						beams.get(i).add(temp);
					}
				}
			
				
			} else {
				// there is a entity ended after i
				if(i==0) {
					PerceptronOutputData1 temp = (PerceptronOutputData1) PerceptronOutputData1.append(null, Perceptron.EMPTY, x, i, i);;
					beams.get(i).add(temp);
				} else {
					for(int k=0;k<beams.get(i-1).size();k++) {
						PerceptronOutputData1 temp = (PerceptronOutputData1)beams.get(i-1).get(k);
						// split the ith segment and put part of it(ended at i) into temp
						for(int j=lastSegment.start; j<=i; j++) {
							temp = (PerceptronOutputData1) PerceptronOutputData1.append(temp, Perceptron.EMPTY, x, j, j);
						}
						beams.get(i).add(temp);
					}
				}
				
				
				
			}
				

			
			PerceptronStatus statusKBest1 = new PerceptronStatus(null, i, 2);
			for(int j=i-1;j>=0;j--) {
				ArrayList<PerceptronOutputData> buf = new ArrayList<PerceptronOutputData>();
				for(int yy=0;yy<beams.get(i).size();yy++) {
					PerceptronOutputData1 yInBeam = (PerceptronOutputData1)beams.get(i).get(yy);
					buf.add(yInBeam);
					// Aligned begin
					Entity entityI = null;
					Entity entityJ = null;
					for(int m=0;m<yInBeam.segments.size();m++) {
						Entity entity = yInBeam.segments.get(m);
						if(entity.type.equals(Perceptron.EMPTY))
							continue;
						if(entity.end == i)
							entityI = entity;
						if(entity.end == j)
							entityJ = entity;
						if(entityI!=null && entityJ!=null)
							break;
					}
					if(entityI!=null && entityJ!=null) {
					// Aligned end
						for(int r=0;r<alphabet2.size();r++) {
							
							// Connect begin
							PerceptronOutputData1 ret = new PerceptronOutputData1(false, -1);
							for(int m=0;m<yInBeam.segments.size();m++) {
								ret.segments.add(yInBeam.segments.get(m));
							}
							for(RelationEntity relation:yInBeam.relations) {
								ret.relations.add(relation);
							}
							RelationEntity relation = new RelationEntity(alphabet2.get(r), entityI, entityJ);
							ret.relations.add(relation);
							buf.add(ret);
							// Connect end
						}
					}
				}

				kBest(x, statusKBest1, beams.get(i), buf, beamSize, other);
			}
			// early update
			if(isTrain) {
				int m=0;
				for(;m<beams.get(i).size();m++) {
					if(beams.get(i).get(m).isIdenticalWith(x, y, statusKBest1)) {
						break;
					}
				}
				if(m==beams.get(i).size() && isAlignedWithGold(beams.get(i).get(0), y, i)) {
					PerceptronStatus returnType = new PerceptronStatus(beams.get(i).get(0), i, 2);
					return returnType;
				}
			}
		}
		
		PerceptronStatus returnType = new PerceptronStatus(beams.get(x.tokens.size()-1).get(0), x.tokens.size()-1, 3);
		return returnType;
	}
	
		
	public PerceptronRelation(ArrayList<String> alphabet2, float convergeThreshold, double weightMax) {
		super(null, alphabet2, convergeThreshold, weightMax);
		
	}
	
	@Override
	public void setFeatureFunction(ArrayList<PerceptronFeatureFunction> featureFunctions1, ArrayList<PerceptronFeatureFunction> featureFunctions2) {
		this.featureFunctions2 = featureFunctions2;
	}
	
	@Override
	public void normalizeWeight() {
		
		double norm2 = w2.twoNorm();
		for(int j=0;j<w2.getIndices().length;j++) {
			w2.setValueAtLocation(j, w2.valueAtLocation(j)/norm2);
		}
	}

	@Override
	public void trainPerceptron(int T, int beamSize, ArrayList<PerceptronInputData> input, ArrayList<PerceptronOutputData> output, Object other) {
		try {		
			for(int i=0;i<T;i++) {
				SparseVector old2 = (SparseVector)w2.cloneMatrix();
				
				long startTime = System.currentTimeMillis();
				//Matrix w_copy = w.copy();
				for(int j=0;j<input.size();j++) {
					PerceptronInputData x = input.get(j);
					PerceptronOutputData y = output.get(j);
					// get the best predicted answer
					PerceptronStatus status = beamSearch(x, y, true, beamSize, other);
					
					if(!status.z.isIdenticalWith(x, y, status)) {
						// if the predicted answer are not identical to the gold answer, update the model.

						if(status.step==2) {
							FReturnType rtFxy = f(x, status, y, other);
							FReturnType rtFxz = f(x, status, status.z, other);
							
														
							SparseVector fxy2  = rtFxy.sv2;
							SparseVector fxz2 = rtFxz.sv2;
							SparseVector temp = fxy2.vectorAdd(fxz2, -1);
							w2 = w2.vectorAdd(temp, 1);
						}
						else if(status.step==3) {
							FReturnType rtFxy = f(x, status, y, other);
							FReturnType rtFxz = f(x, status, status.z, other);
							
							
							
							SparseVector fxy2  = rtFxy.sv2;
							SparseVector fxz2 = rtFxz.sv2;
							SparseVector temp = fxy2.vectorAdd(fxz2, -1);
							w2 = w2.vectorAdd(temp, 1);
						} else
							throw new Exception();
						
						
					}
				}
				// check weight
				if(weightMax!=0) {
					
					double values2[] = w2.getValues();
					for(int j=0;j<values2.length;j++) {
						if(values2[j]>weightMax)
							values2[j] = weightMax;
						else if(values2[j]<-weightMax)
							values2[j] = -weightMax;
					}
				}
				

				
				float dist2 = (float)old2.vectorAdd(w2, -1).twoNorm();
				float dist = dist2;
				
				
				if(dist<convergeThreshold) {
					System.out.println("converged, quit training");
					normalizeWeight();

					return;
				} 
				else {
					long endTime = System.currentTimeMillis();
					System.out.println((i+1)+" train finished "+(dist2)+" "+(endTime-startTime)+" ms");
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
			if(status.step==2) {
				scores.add(w2.dotProduct(ret.sv2));
			} else 
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
	
	
	@Override
	protected FReturnType f(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other) throws Exception {	
		
		if(y.isGold) {
			if(status.step==0) { // initialize the feature vectors of gold output
								
				TObjectDoubleHashMap<String> map2 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions2.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions2.get(j);
					featureFunction.compute(x, status, y, other, map2);
				}
				y.featureVectors2.add(hashMapToSparseVector(map2));
				return new FReturnType(null, y.featureVectors2.get(status.tokenIndex));
			} else if(status.step==2) {
				return new FReturnType(null, y.featureVectors2.get(status.tokenIndex));
			} else if(status.step==3) {
				return new FReturnType(null, y.featureVectors2.get(status.tokenIndex));
			} else
				throw new Exception();
			
		} else {
			if(status.step==2) {
								
				TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions2.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions2.get(j);
					featureFunction.compute(x, status, y, other, map);
				}
				y.featureVector2 = hashMapToSparseVector(map);
				return new FReturnType(null, y.featureVector2);
			} else if(status.step==3) {
								
				TObjectDoubleHashMap<String> map2 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions2.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions2.get(j);
					featureFunction.compute(x, status, y, other, map2);
				}
				y.featureVector2 = hashMapToSparseVector(map2);
				return new FReturnType(null, y.featureVector2);
			} else
				throw new Exception();
			
		}

	}
	
	
	
	public static boolean isAlignedWithGold(PerceptronOutputData predict, PerceptronOutputData gold, int tokenIndex) {
		PerceptronOutputData1 predict1 = (PerceptronOutputData1)predict;
		PerceptronOutputData1 gold1 = (PerceptronOutputData1)gold;
		
		Entity predictLastSeg = predict1.getLastSegment(tokenIndex);
		Entity goldLastSeg = gold1.getLastSegment(tokenIndex);
		if(predictLastSeg.end==goldLastSeg.end)
			return true;
		else return false;
	}
	
}
