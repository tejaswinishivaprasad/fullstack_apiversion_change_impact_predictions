package org.impactframework.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ImpactItemDTO {

    private String service;

    @JsonProperty("risk_score")
    private double riskScore;


    public String getService() { return service; }
    public void setService(String service) { this.service = service; }

    public double getRiskScore() { return riskScore; }
    public void setRiskScore(double riskScore) { this.riskScore = riskScore; }
}
