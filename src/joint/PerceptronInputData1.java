package joint;

import gnu.trove.TIntArrayList;

import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.nlp.Sentence;

public class PerceptronInputData1 extends PerceptronInputData{

	private static final long serialVersionUID = -8439560686607969171L;
	public TIntArrayList offset; // the offset that each token start at and the first offset may not be 0 
	public Sentence sentInfo; // the information(pos, syntactic, dep) about the sentence which denotes this PerceptronInputData
	
	
	public PerceptronInputData1() {
		offset = new TIntArrayList();
	}
	
	@Override
	public String toString() {
		String s = "words: "+tokens.get(0);
		for(int i=1;i<tokens.size();i++)
			s += ", "+tokens.get(i);
		s+="\n";
		return s;
	}
	
}