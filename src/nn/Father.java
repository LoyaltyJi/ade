package nn;

import java.util.List;

public abstract class Father {
	public abstract double[][] getE();
	public abstract int getPaddingID();
	public abstract int getPositionID(int position);
	public abstract double[][] getEg2E();
	public abstract List<String> getKnownWords();
}
