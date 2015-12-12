package sparse_pipeline;



class GradientKeeper {
	double[][] gradW;
	double[] gradB;
	
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public GradientKeeper(Parameters parameters, SparseLayer sparse) {
	    gradW = new double[sparse.W.length][sparse.W[0].length];
	    
	    if(sparse.useB)
	    	gradB = new double[sparse.B.length];
	}
}