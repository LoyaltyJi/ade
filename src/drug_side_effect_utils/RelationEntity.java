package drug_side_effect_utils;

import java.io.Serializable;

public class RelationEntity implements Serializable{

	private static final long serialVersionUID = 6447119639004030919L;
	public String type;
	public Entity entity1; 
	public Entity entity2; 
	public RelationEntity(String type, Entity entity1, Entity entity2) {
		super();
		this.type = type;
		this.entity1 = entity1;
		this.entity2 = entity2;
	}
	
	// Judge whether "a" belongs to this relation.
	// if not, return null; else , return the other entity.
	public Entity hasEntity(Entity a) {
		Entity b = null;
		if(entity1.equals(a))
			b=entity2;
		else if(entity2.equals(a))
			b = entity1;
		return b;
	}
	
	public Entity getFormer() {
		Entity former = entity1.offset>entity2.offset ? entity2:entity1;
		return former;
	}
	
	public Entity getLatter() {
		Entity latter = entity1.offset>entity2.offset ? entity1:entity2;
		return latter;
	}
	
	/*
	 * equals and hashCode make sure that the relation is non-directed.
	 */
	@Override
	public boolean equals(Object obj) {
		if(obj == null || !(obj instanceof RelationEntity))
			return false;
		RelationEntity o = (RelationEntity)obj;
		if(type.equals(o.type)) {
			if(entity1.equals(o.entity1) && entity2.equals(o.entity2))
				return true;
			else if(entity1.equals(o.entity2) && entity2.equals(o.entity1))
				return true;
			else 
				return false;
		} else
			return false;
		
		
	}
	
	@Override
	public int hashCode() {
	    return entity1.hashCode()+entity2.hashCode()+type.hashCode();  
		
	}
	
	@Override
	public String toString() {
		return entity1+", "+entity2+", "+type;
	}
}
