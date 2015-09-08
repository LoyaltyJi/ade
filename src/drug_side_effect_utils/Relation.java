package drug_side_effect_utils;

import java.io.Serializable;

public class Relation implements Serializable{

	private static final long serialVersionUID = 1205844430364130631L;
	public String id;
	public String type;
	public String mesh1; 
	public String mesh2; 
	public Relation(String id, String type, String mesh1, String mesh2) {
		super();
		this.id = id;
		this.type = type;
		this.mesh1 = mesh1;
		this.mesh2 = mesh2;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj == null || !(obj instanceof Relation))
			return false;
		Relation o = (Relation)obj;
		
		if(o.type.equals(this.type)) {
			if(o.mesh1.equals(this.mesh1) && o.mesh2.equals(this.mesh2))
				return true;
			else if(o.mesh1.equals(this.mesh2) && o.mesh2.equals(this.mesh1))
				return true;
			else 
				return false;
		} else 
			return false;
	}
	
	@Override
	public int hashCode() {
	    return type.hashCode()+mesh1.hashCode()+mesh2.hashCode();  
	}
	
	@Override
	public String toString() {
		return mesh1+" "+mesh2+" "+type;
	}
}
