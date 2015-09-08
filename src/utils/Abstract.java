package utils;

import java.io.Serializable;
import java.util.HashSet;

public class Abstract implements Serializable{

	private static final long serialVersionUID = -6363911384128572388L;
	public String id;
	public HashSet<ADESentence> sentences; // the sentences have no order
	
	public Abstract() {
		sentences = new HashSet<>();
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj == null || !(obj instanceof Abstract))
			return false;
		Abstract o = (Abstract)obj;
		if(o.id.equals(this.id))
			return true;
		else 
			return false;
	}
	
	@Override
	public int hashCode() {
		
	    return id.hashCode();  
	}
	
	@Override
	public String toString() {
		return id;
	}
}
