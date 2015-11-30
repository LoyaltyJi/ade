package nn;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import cc.mallet.types.SparseVector;
import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;
import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;

public class Prediction implements Comparable<Prediction>{
	public List<Entity> entities;
	public List<RelationEntity> relations;
	public TIntArrayList labels;
	public TDoubleArrayList scores;
	
	// perceptron
	//HashSet<String> currentFeatures;
	HashSet<String> entityFeatures;
	HashSet<String> relationFeatures;
	Perceptron perceptron;
	
	public Prediction() {
		entities = new ArrayList<>();
		relations = new ArrayList<>();
		labels = new TIntArrayList();
		scores = new TDoubleArrayList();
		entityFeatures = new HashSet<>();
		relationFeatures = new HashSet<>();
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
			scores.add(other.scores.get(i));
		}
		
		entityFeatures.addAll(other.entityFeatures);
		relationFeatures.addAll(other.relationFeatures);
	}
	
	public void addLabel(int label, double score) {
		labels.add(label);
		scores.add(score);
	}
	
	
	@Override
	public int compareTo(Prediction o) {

		/*SparseVector otherVector = perceptron.featuresToSparseVector(o.currentFeatures);
		double otherScore = perceptron.w.dotProduct(otherVector);
		SparseVector vector = perceptron.featuresToSparseVector(this.currentFeatures);
		double score = perceptron.w.dotProduct(vector);*/
		SparseVector otherVector = null;
		if(o.relationFeatures.isEmpty()) {
			otherVector = perceptron.featuresToSparseVector(o.entityFeatures);
		} else {
			HashSet<String> tempFeatures = new HashSet<>();
			tempFeatures.addAll(o.entityFeatures);
			tempFeatures.addAll(o.relationFeatures);
			otherVector = perceptron.featuresToSparseVector(tempFeatures);
		}
		double otherScore = perceptron.w.dotProduct(otherVector);
		
		SparseVector vector = null;
		if(this.relationFeatures.isEmpty()) {
			vector = perceptron.featuresToSparseVector(this.entityFeatures);
		} else {
			HashSet<String> tempFeatures = new HashSet<>();
			tempFeatures.addAll(this.entityFeatures);
			tempFeatures.addAll(this.relationFeatures);
			vector = perceptron.featuresToSparseVector(tempFeatures);
		}
		double score = perceptron.w.dotProduct(vector);
		
		/*double otherScore = 0;
		for(String other:o.currentFeatures) {
			int index = perceptron.getFeatureIndex(other);
			if(index == -1)
				continue;
			if(perceptron.w[index]==0)
				continue;
			otherScore += perceptron.w[index];
		}
		double score = 0;
		for(String thiss:this.currentFeatures) {
			int index = perceptron.getFeatureIndex(thiss);
			if(index == -1)
				continue;
			if(perceptron.w[index]==0)
				continue;
			score += perceptron.w[index];
		}*/
		
		/*double otherScore = 0;
		for(int i=0;i<o.scores.size();i++) {
			otherScore += o.scores.get(i);
		}
		double score = 0;
		for(int i=0;i<this.scores.size();i++) {
			score += this.scores.get(i);
		}*/
		
		

		if(score < otherScore)
			return 1;
		else if(score > otherScore)
			return -1;
		else
			return 0;
	}
}
