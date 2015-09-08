package joint;

import gnu.trove.TIntArrayList;

import java.util.ArrayList;

import cc.mallet.types.SparseVector;
import cn.fox.machine_learning.Perceptron;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;

public class Perceptron1 extends Perceptron{

	private static final long serialVersionUID = -503123511947042917L;
	// the maximum length of each alphabet1 type
	private TIntArrayList d;
	
	public Perceptron1(ArrayList<String> alphabet1, ArrayList<String> alphabet2, TIntArrayList d, float convergeThreshold, double weightMax) {
		super(alphabet1, alphabet2, convergeThreshold, weightMax);
		
		this.d = new TIntArrayList();
		this.d.add(1);
		for(int i=0;i<d.size();i++) {
			this.d.add(d.get(i));
		}

		this.alphabet1.add(0, Perceptron.EMPTY);
	
	}
	
	@Override
	public void buildFeatureAlphabet(ArrayList<PerceptronInputData> inputDatas, ArrayList<PerceptronOutputData> outputDatas, Object other) {
		try {
			w1 = new SparseVector();
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
			ArrayList<PerceptronOutputData> buf = new ArrayList<PerceptronOutputData>();
			for(int t=0;t<this.alphabet1.size();t++) {
				if(i==0) {
					buf.add(PerceptronOutputData1.append(null, alphabet1.get(t), x, 0, 0));
					continue;
				}
				for(int dd=1;dd<=this.d.get(t);dd++) {
					if(i-dd>=0) {
						for(int yy=0;yy<beams.get(i-dd).size();yy++) {
							int k = i-dd+1;
							buf.add(PerceptronOutputData1.append(beams.get(i-dd).get(yy), alphabet1.get(t), x, k, i));
						}
					} else if(i-dd==-1){ 
						buf.add(PerceptronOutputData1.append(null, alphabet1.get(t), x, 0, i));
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

			PerceptronStatus statusKBest1 = new PerceptronStatus(null, i, 2);
			for(int j=i-1;j>=0;j--) {
				buf.clear();
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
