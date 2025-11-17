package org.impactframework.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

public class ReportDTO {

    private String dataset;

    @JsonProperty("old_file")
    private String oldFile;

    @JsonProperty("new_file")
    private String newFile;

    @JsonProperty("risk_score")
    private Double riskScore;             

    @JsonProperty("risk_band")
    private String riskBand;

    private String summary;
    private List<DiffItemDTO> details;

    @JsonProperty("ai_explanation")
    private String aiExplanation;

    private Map<String, Object> backend;
    private List<String> logs;

    @JsonProperty("backend_impacts")
    private List<ImpactItemDTO> backendImpacts;

    @JsonProperty("frontend_impacts")
    private List<ImpactItemDTO> frontendImpacts;

    @JsonProperty("predicted_risk")
    private Double predictedRisk;         

    private Map<String, Double> confidence;
    private Map<String, Object> versioning;


    public String getDataset() { return dataset; }
    public void setDataset(String dataset) { this.dataset = dataset; }

    public String getOldFile() { return oldFile; }
    public void setOldFile(String oldFile) { this.oldFile = oldFile; }

    public String getNewFile() { return newFile; }
    public void setNewFile(String newFile) { this.newFile = newFile; }

    public Double getRiskScore() { return riskScore; }
    public void setRiskScore(Double riskScore) { this.riskScore = riskScore; }

    public String getRiskBand() { return riskBand; }
    public void setRiskBand(String riskBand) { this.riskBand = riskBand; }

    public String getSummary() { return summary; }
    public void setSummary(String summary) { this.summary = summary; }

    public List<DiffItemDTO> getDetails() { return details; }
    public void setDetails(List<DiffItemDTO> details) { this.details = details; }

    public String getAiExplanation() { return aiExplanation; }
    public void setAiExplanation(String aiExplanation) { this.aiExplanation = aiExplanation; }

    public Map<String, Object> getBackend() { return backend; }
    public void setBackend(Map<String, Object> backend) { this.backend = backend; }

    public List<String> getLogs() { return logs; }
    public void setLogs(List<String> logs) { this.logs = logs; }

    public List<ImpactItemDTO> getBackendImpacts() { return backendImpacts; }
    public void setBackendImpacts(List<ImpactItemDTO> backendImpacts) { this.backendImpacts = backendImpacts; }

    public List<ImpactItemDTO> getFrontendImpacts() { return frontendImpacts; }
    public void setFrontendImpacts(List<ImpactItemDTO> frontendImpacts) { this.frontendImpacts = frontendImpacts; }

    public Double getPredictedRisk() { return predictedRisk; }
    public void setPredictedRisk(Double predictedRisk) { this.predictedRisk = predictedRisk; }

    public Map<String, Double> getConfidence() { return confidence; }
    public void setConfidence(Map<String, Double> confidence) { this.confidence = confidence; }

    public Map<String, Object> getVersioning() { return versioning; }
    public void setVersioning(Map<String, Object> versioning) { this.versioning = versioning; }
}
