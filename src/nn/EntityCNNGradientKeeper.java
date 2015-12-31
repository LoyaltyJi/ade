package nn;

public class EntityCNNGradientKeeper {
	public double[][] gradW;
	public double[] gradB;
	public double[][] gradE;
	
	
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public EntityCNNGradientKeeper(Parameters parameters, EntityCNN cnn, GradientKeeper gk) {
		gradW = new double[cnn.filterW.length][cnn.filterW[0].length];
		gradB = new double[cnn.filterB.length];
		gradE = gk.gradE;
			
	}
	
	public EntityCNNGradientKeeper(Parameters parameters, EntityCNN cnn, GradientKeeper1 gk) {
		gradW = new double[cnn.filterW.length][cnn.filterW[0].length];
		gradB = new double[cnn.filterB.length];
		gradE = gk.gradE;
			
	}
	
	public EntityCNNGradientKeeper(CombineParameters parameters, CombineEntityCNN cnn, GradientKeeper1 gk) {
		gradW = new double[cnn.filterW.length][cnn.filterW[0].length];
		gradB = new double[cnn.filterB.length];
		gradE = gk.gradE;
			
	}
}
