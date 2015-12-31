package nn;

public class GradientKeeper1 {
	public double[][] gradWh;
	public double[] gradBh;
	public double[][] gradWo;
	public double[][] gradE;
	public EntityCNNGradientKeeper entityKeeper;
	public SentenceCNNGradientKeeper sentenceKeeper;
	
		
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public GradientKeeper1(Parameters parameters, NNSimple nn) {
		gradWh = new double[nn.Wh.length][nn.Wh[0].length];
		
		gradBh = new double[nn.Bh.length];
		
		
		gradWo = new double[nn.Wo.length][nn.Wo[0].length];
		gradE = new double[nn.owner.getE().length][nn.owner.getE()[0].length];
			
	}
	
	public GradientKeeper1(CombineParameters parameters, CombineNN nn) {
		gradWh = new double[nn.Wh.length][nn.Wh[0].length];
		
		gradBh = new double[nn.Bh.length];
		
		
		gradWo = new double[nn.Wo.length][nn.Wo[0].length];
		gradE = new double[nn.owner.getE().length][nn.owner.getE()[0].length];
			
	}
}
