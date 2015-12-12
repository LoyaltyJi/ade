package sparse_pipeline;

import java.util.HashSet;
import java.util.List;

import drug_side_effect_utils.Entity;
import edu.stanford.nlp.ling.CoreLabel;
import gnu.trove.TIntArrayList;


class Example {
	/*
	 *  Indicate whether it's a relation example
	 */
	public boolean bRelation = false;
	/*
	 * the id of features
	 */
	public TIntArrayList featureIdx;

  public double[] label;

  public Example(boolean bRelation) {
    this.bRelation = bRelation;
    
    this.featureIdx = new TIntArrayList();

  }

  
}
