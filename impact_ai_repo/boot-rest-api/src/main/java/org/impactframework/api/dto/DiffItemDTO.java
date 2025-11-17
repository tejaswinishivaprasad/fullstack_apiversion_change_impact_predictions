package org.impactframework.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class DiffItemDTO {

    private String type;
    private String path;
    private String method;
    private String detail;


    public String getType() { return type; }
    public void setType(String type) { this.type = type; }

    public String getPath() { return path; }
    public void setPath(String path) { this.path = path; }

    public String getMethod() { return method; }
    public void setMethod(String method) { this.method = method; }

    public String getDetail() { return detail; }
    public void setDetail(String detail) { this.detail = detail; }
}
