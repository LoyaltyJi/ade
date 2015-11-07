package nn;

import java.util.List;

import gnu.trove.TIntArrayList;


class Example {
	/*
	 *  Indicate whether it's a relation example
	 */
	public boolean bRelation = false;
	/*
	 * It includes the row indexes of the embedding matrix (all the non-composite features).
	 * -1 denotes the feature is disabled
	 */
	public TIntArrayList featureIdx;
		
  /*
   * It denotes the composite features, such as a entity feature which we need to composite all 
   * the embedding features related to that entity by pooling or convolution.
   * These features should be put on the last of the input layer.
   * empty denotes the feature is disabled.
   */
  public TIntArrayList formerIdx;
  public TIntArrayList latterIdx;
  
  public TIntArrayList sentenceIdx;
  public TIntArrayList positionIdxFormer;
  public TIntArrayList positionIdxLatter;
  
  
  
  /*
   * The label vector of this example
   * e.g. 0,0,1,0,0
   */
  public TIntArrayList label;

  public Example(boolean bRelation) {
    this.bRelation = bRelation;
    
    this.featureIdx = new TIntArrayList();
    this.formerIdx = new TIntArrayList();
    this.latterIdx = new TIntArrayList();
    this.sentenceIdx = new TIntArrayList();
    this.positionIdxFormer = new TIntArrayList();
    this.positionIdxLatter = new TIntArrayList();
  }

  
}
