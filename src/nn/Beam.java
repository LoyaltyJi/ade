package nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Beam {
	public List<Prediction> items;
	public int size;

	
	public Beam(int size) {
		items = new ArrayList<Prediction>();
		this.size = size;

	}
	
	public void kbest(List<Prediction> toBeAdd) {
		Collections.sort(toBeAdd);
		this.items.clear();
		int min = toBeAdd.size()>size ? size: toBeAdd.size();
		for(int i=0;i<min;i++) {
			this.items.add(toBeAdd.get(i));
		}
	}
}
