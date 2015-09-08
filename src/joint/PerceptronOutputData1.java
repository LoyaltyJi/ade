package joint;

import java.util.ArrayList;
import java.util.HashSet;

import cn.fox.machine_learning.PerceptronInputData;
import cn.fox.machine_learning.PerceptronOutputData;
import cn.fox.machine_learning.PerceptronStatus;
import cn.fox.machine_learning.Perceptron;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;

public class PerceptronOutputData1 extends PerceptronOutputData{

	private static final long serialVersionUID = 2352008110784650763L;
	// each entity denote a segment because a segment may not be a entity.
	// the last segment of ArrayList may not be the currentSegment, because this object can be gold
	// the currentSegment must end at the tokenIndex(in the PerceptronStatus) token.
	public ArrayList<Entity> segments; 
	public HashSet<RelationEntity> relations;

	
	
	
	public PerceptronOutputData1(boolean isGold, int tokenNumber) {
		super(isGold, tokenNumber);
		segments = new ArrayList<Entity>();
		relations = new HashSet<RelationEntity>();
	}
	
	
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
		PerceptronOutputData1 other1 = (PerceptronOutputData1)other;
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

		if(status.step==2 || status.step==3) {
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
			
			HashSet<RelationEntity> otherRelation = new HashSet<RelationEntity>(); // other is a gold output data
			for(RelationEntity relation:other1.relations) {
				if(relation.entity1.end<=status.tokenIndex && relation.entity2.end<=status.tokenIndex)
					otherRelation.add(relation);
			}
			if(!relations.equals(otherRelation))
				return false;
			
		}
		
		return true;
	}
	
	public static PerceptronOutputData append(PerceptronOutputData yy, String t, PerceptronInputData xx, int k, int i) {
		PerceptronInputData1 x = (PerceptronInputData1)xx;
		PerceptronOutputData1 y = (PerceptronOutputData1)yy;
		PerceptronOutputData1 ret = new PerceptronOutputData1(false, -1);
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
		// copy relation
		for(RelationEntity relation:y.relations) {
			ret.relations.add(relation);
		}
		return ret;
	}
	
		
	
	@Override
	public String toString() {
		String s = "segments: "+segments.get(0);
		for(int i=1;i<segments.size();i++)
			s += ", "+segments.get(i);
		s+="\n";
		s+="relations: ";
		int i=0;
		for(RelationEntity relation:relations) {
			if(i!=0)
				s+=", "+relation;
			else
				s+=relation;
			i++;
		}
		s+="\n";
		return s;
	}
}
