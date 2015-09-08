package pipeline;

import java.util.ArrayList;

import joint.PerceptronInputData1;
import joint.PerceptronOutputData1;
import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import drug_side_effect_utils.Entity;

public class PerceptronEntityOutputData extends PerceptronOutputData1 {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8531622346426337351L;
	
	
	public PerceptronEntityOutputData(boolean isGold, int tokenNumber) {
		super(isGold, tokenNumber);
		segments = new ArrayList<Entity>();
	}

	@Override
	public Entity getLastSegment(int tokenIndex) {
		if(isGold) {
			int i=0;
			Entity thisSegment = null;
			do {
				thisSegment = segments.get(i);
				i++;
			}while(tokenIndex>thisSegment.end);
			return thisSegment;
		} else {
			return segments.get(segments.size()-1);
		}
	}
	@Override
	public int getLastSegmentIndex(int tokenIndex) {
		if(isGold) {
			int i=0;
			Entity thisSegment = null;
			do {
				thisSegment = segments.get(i);
				i++;
			}while(tokenIndex>thisSegment.end);
			return i-1;
		} else {
			return segments.size()-1;
		}
	}
	
	@Override
	public boolean isIdenticalWith(PerceptronInputData input, PerceptronOutputData other, PerceptronStatus status) {
		PerceptronEntityOutputData other1 = (PerceptronEntityOutputData)other;
		if(status.step==1) {
			int i=0;
			Entity thisSegment = null;
			Entity OtherSegment = null;
			do {
				thisSegment = segments.get(i);
				OtherSegment = other1.segments.get(i);
				if(!thisSegment.equals(OtherSegment))
					return false;
				
				i++;
			}while(status.tokenIndex>thisSegment.end);
			
			return true;
		} 

		if(status.step==3) {
			int i=0;
			Entity thisSegment = null;
			Entity OtherSegment = null;
			do {
				thisSegment = segments.get(i);
				OtherSegment = other1.segments.get(i);
				if(!thisSegment.equals(OtherSegment))
					return false;
				
				i++;
			}while(status.tokenIndex>thisSegment.end);
			
			
		}
		
		return true;
	}
	
	
	public static PerceptronOutputData append(PerceptronOutputData yy, String t, PerceptronInputData xx, int k, int i) {
		PerceptronInputData1 x = (PerceptronInputData1)xx;
		PerceptronEntityOutputData y = (PerceptronEntityOutputData)yy;
		PerceptronEntityOutputData ret = new PerceptronEntityOutputData(false, -1);
		if(yy == null) {
			// append segment
			int segmentOffset = x.offset.get(k);
			String segmentText = "";
			for(int m=k;m<=i;m++) {
				int whitespaceToAdd = x.offset.get(m)-(segmentOffset+segmentText.length());
				if(whitespaceToAdd>0) {
					for(int j=0;j<whitespaceToAdd;j++)
						segmentText += " ";
				}
				segmentText += x.tokens.get(m);
			}
			Entity segment = new Entity(null, t, segmentOffset, segmentText, null);
			segment.start = k;
			segment.end = i;
			ret.segments.add(segment);
			return ret;
		}
			
		// copy segment
		for(int m=0;m<y.segments.size();m++) {
			ret.segments.add(y.segments.get(m));
		}
		// append segment
		int segmentOffset = x.offset.get(k);
		String segmentText = "";
		for(int m=k;m<=i;m++) {
			int whitespaceToAdd = x.offset.get(m)-(segmentOffset+segmentText.length());
			if(whitespaceToAdd>0) {
				for(int j=0;j<whitespaceToAdd;j++)
					segmentText += " ";
			}	
			segmentText += x.tokens.get(m);
		}
		Entity segment = new Entity(null, t, segmentOffset, segmentText, null);
		segment.start = k;
		segment.end = i;
		ret.segments.add(segment);
		
		return ret;
	}
	
	
	
	@Override
	public String toString() {
		String s = "segments: "+segments.get(0);
		for(int i=1;i<segments.size();i++)
			s += ", "+segments.get(i);
		s+="\n";
		
		return s;
	}
}
