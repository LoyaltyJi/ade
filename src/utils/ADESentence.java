package utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;

import drug_side_effect_utils.Entity;
import drug_side_effect_utils.RelationEntity;

public class ADESentence implements Serializable{

	private static final long serialVersionUID = -5182216844953339780L;
	public int offset; // sentence offset based on the document
	public String text;
	public HashSet<Entity> entities; // use hashset to get rid of the overlapped entities
	public HashSet<RelationEntity> relaitons;
	
	public ADESentence() {
		entities = new HashSet<>();
		relaitons = new HashSet<>();
	}
	
	public ADESentence(String text) {
		this.text =text;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj == null || !(obj instanceof ADESentence))
			return false;
		ADESentence o = (ADESentence)obj;
		if(o.text.equals(this.text))
			return true;
		else 
			return false;
	}
	
	@Override
	public int hashCode() {
		
	    return text.hashCode();  
	}
	
	@Override
	public String toString() {
		return text;
	}
}
