package sparse_pipeline;

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

		
		
		

		return 0;
	}
}
