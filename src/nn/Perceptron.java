package nn;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;

import cc.mallet.types.SparseVector;
import drug_side_effect_utils.Entity;
import edu.stanford.nlp.ling.CoreLabel;
import gnu.trove.TObjectIntHashMap;

public class Perceptron implements Serializable {
	//double[] w; 
	SparseVector w;
	HashSet<String> featureAlphabet;
	private TObjectIntHashMap<String> featureMap;
	boolean freezeFeature;
	
	public Perceptron() {
		//w = new SparseVector();
		featureAlphabet = new HashSet<>();
		featureMap = new TObjectIntHashMap<>();
		freezeFeature = false;
	}
	
	/*
	 * This is only the gold features after preprocessing.
	 * When the training begins, some features will be added dynamically.
	 * But when test begins, freeze the feature alphabet.
	 */
	public void addGoldFeature() {
		int m = 0;
	    for (String temp : featureAlphabet)
	    	featureMap.put(temp, (m++));
	    
	    //w = new double[featureAlphabet.size()];
	    w = new SparseVector();
	}
	
	public int getFeatureIndex(String featureName) {
		if(freezeFeature) {
			if(featureAlphabet.contains(featureName)) {
				return featureMap.get(featureName);
			} else {
				return -1;
			}
		} else {
			if(featureAlphabet.contains(featureName)) {
				return featureMap.get(featureName);
			} else {
				featureAlphabet.add(featureName);
				featureMap.put(featureName, featureAlphabet.size());
				return featureAlphabet.size();
			}
		}
		
	}
		
	public SparseVector featuresToSparseVector(HashSet<String> features) {
		int[] featureIndicesArr = new int[features.size()];
        double[] featureValuesArr = new double[features.size()];
        int count = 0;
		for(String featureName:features) {
			int index = getFeatureIndex(featureName);
			if(index == -1)
				continue;
			featureIndicesArr[count] = index;
        	featureValuesArr[count] = 1;
        	count++;
		}
		
        SparseVector fxy = new SparseVector(featureIndicesArr, featureValuesArr, false);
		
		
        return fxy;
	}
	
	public HashSet<String> featureFunction(List<CoreLabel> tokens, int idx, String transitionName, Entity drug, Entity disease) {
		HashSet<String> features = new HashSet<>();
		
		if(idx != -1) {
			features.add("TK#"+tokens.get(idx).lemma()+"#"+transitionName);
		} else {
			features.add(drug.text.toLowerCase()+"#"+disease.text.toLowerCase()+"#"+transitionName);
		}
		
		return features;
	}
	
	
}
