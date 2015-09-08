package utils;

import java.io.Serializable;
import java.util.HashMap;

import cc.mallet.types.Instance;
import drug_side_effect_utils.Entity;

// it denotes the a true/false relation between ae and drug.
public class ADELine implements Serializable {

	private static final long serialVersionUID = -4190744599787549380L;
	public String id;
	public String sentence;
	public Entity ae;
	public Entity drug;
	public HashMap<String,Double> map;
	public boolean positive;
	public ADELine(String id, String sentence, Entity ae, Entity drug) {
		super();
		this.id = id;
		this.sentence = sentence;
		this.ae = ae;
		this.drug = drug;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj == null || !(obj instanceof ADELine))
			return false;
		ADELine o = (ADELine)obj;
		if(o.ae!=null && ae!=null && o.drug!=null && drug!=null) {
			if(o.id.equals(this.id) && o.sentence.equals(this.sentence) && o.ae.equals(ae) && o.drug.equals(drug))
				return true;
			else 
				return false;
		} else {
			if(o.id.equals(this.id) && o.sentence.equals(this.sentence))
				return true;
			else 
				return false;
		}
	}
	
	@Override
	public int hashCode() {
		if(ae!=null && drug!=null) {
			return id.hashCode()+sentence.hashCode()+ae.hashCode()+drug.hashCode();  
		} else {
			return id.hashCode()+sentence.hashCode();  
		}
	    
	}
	
	@Override
	public String toString() {
		return sentence.toString();
	}
}
