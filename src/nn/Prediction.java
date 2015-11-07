package nn;

import java.util.ArrayList;
import java.util.List;

import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;
import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;

public class Prediction implements Comparable<Prediction>{
	public List<Entity> entities;
	public List<RelationEntity> relations;
	public TIntArrayList labels;
	public TDoubleArrayList scores;
	public double sumScore;
	public boolean beamMeanSoftmax;
	
	public Prediction(boolean beamMeanSoftmax) {
		entities = new ArrayList<>();
		relations = new ArrayList<>();
		labels = new TIntArrayList();
		scores = new TDoubleArrayList();
		this.beamMeanSoftmax = beamMeanSoftmax;
	}
	
	public void copy(Prediction other) {
		for(int i=0;i<other.labels.size();i++) {
			labels.add(other.labels.get(i));
		}
		for(int i=0;i<other.entities.size();i++) {
			entities.add(other.entities.get(i));
		}
		for(int i=0;i<other.relations.size();i++) {
			relations.add(other.relations.get(i));
		}
		for(int i=0;i<other.scores.size();i++) {
			scores.add(other.scores.size());
		}
		sumScore = other.sumScore;
	}
	
	public void addLabel(int label, double score) {
		labels.add(label);
		scores.add(score);
		sumScore += score;
	}
	
	
	@Override
	public int compareTo(Prediction o) {
		if(beamMeanSoftmax) {
			double avgThis = 0;
			for(int i=0;i<scores.size();i++) {
				avgThis += scores.get(i)/scores.size();
			}
			double avgOther = 0;
			for(int i=0;i<o.scores.size();i++) {
				avgOther += o.scores.get(i)/o.scores.size();
			}
			if(avgThis < avgOther)
				return 1;
			else if(avgThis > avgOther)
				return -1;
			else
				return 0;
		} else {
			if(sumScore < o.sumScore)
				return 1;
			else if(sumScore > o.sumScore)
				return -1;
			else
				return 0;
		}
		
		
	}
}
