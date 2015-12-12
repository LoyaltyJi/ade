package sparse_pipeline;

import cn.fox.utils.Evaluater;

class DecodeStatistic {
	public int wrong; // wrong transition
	public int total; // total transition
	
	public int ctPredictEntity = 0;
	public int ctTrueEntity = 0;
	public int ctCorrectEntity = 0;
	public int ctPredictRelation = 0;
	public int ctTrueRelation = 0;
	public int ctCorrectRelation = 0;
	
	public double getWrongRate() {
		return wrong*1.0/total;
	}
	
	public double getEntityPrecision() {
		return Evaluater.getPrecisionV2(ctCorrectEntity, ctPredictEntity);
	}
	
	public double getEntityRecall() {
		return Evaluater.getRecallV2(ctCorrectEntity, ctTrueEntity);
	}
	
	public double getEntityF1() {
		return Evaluater.getFMeasure(getEntityPrecision(), getEntityRecall(), 1);
	}
	
	public double getRelationPrecision() {
		return Evaluater.getPrecisionV2(ctCorrectRelation, ctPredictRelation);
	}
	
	public double getRelationRecall() {
		return Evaluater.getRecallV2(ctCorrectRelation, ctTrueRelation);
	}
	
	public double getRelationF1() {
		return Evaluater.getFMeasure(getRelationPrecision(), getRelationRecall(), 1);
	}
}