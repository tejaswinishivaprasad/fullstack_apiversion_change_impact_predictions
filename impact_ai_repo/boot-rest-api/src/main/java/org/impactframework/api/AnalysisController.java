package org.impactframework.api;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.web.util.UriComponentsBuilder;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Collections;



import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.HttpHeaders;


import org.springframework.core.io.Resource;
import org.springframework.core.io.InputStreamResource;
import java.io.InputStream;

@RestController
@RequestMapping("/analysis")
public class AnalysisController {

    @Value("${AI_CORE_URL:http://localhost:8000}")
    private String aiCoreUrl;

    private final RestTemplate rest;

    public AnalysisController(RestTemplate rest) {
        this.rest = rest;
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        String url = aiCoreUrl + "/health";
        try {
            String body = rest.getForObject(url, String.class);
            return ResponseEntity.ok(body);
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("{\"error\":\"AI Core unavailable\"}");
        }
    }

    // Forward GET /analysis/report to AI Core /report and return the raw bytesx
    @GetMapping("/report")
    public ResponseEntity<byte[]> report(
            @RequestParam(name= "old") String old,
            @RequestParam(name = "new") String newSpec,
            @RequestParam(name = "dataset", defaultValue = "openapi") String dataset,
            @RequestParam(required = false) String pairId
    ) {
        System.out.println("inside report");
        StringBuilder url = new StringBuilder(aiCoreUrl + "/report?old=")
                .append(URLEncoder.encode(old, StandardCharsets.UTF_8))
                .append("&new=").append(URLEncoder.encode(newSpec, StandardCharsets.UTF_8))
                .append("&dataset=").append(URLEncoder.encode(dataset, StandardCharsets.UTF_8));
        if (pairId != null && !pairId.isEmpty()) {
            url.append("&pair_id=").append(URLEncoder.encode(pairId, StandardCharsets.UTF_8));
        }

        System.out.println("[AnalysisController] ENTER report() -> forwarding to: " + url);
        long t0 = System.nanoTime();
        try {
            // Use byte[] to avoid RestTemplate/Jackson auto-deserialize surprises
            ResponseEntity<byte[]> resp = rest.exchange(url.toString(), HttpMethod.GET, null, byte[].class);

            long t1 = System.nanoTime();
            double elapsedMs = (t1 - t0) / 1_000_000.0;
            System.out.println("[AnalysisController] AI Core responded status=" + resp.getStatusCodeValue() + " elapsed_ms=" + elapsedMs + " bodyLen=" + (resp.getBody()==null?0:resp.getBody().length));

            HttpHeaders outHeaders = new HttpHeaders();
            MediaType ct = resp.getHeaders().getContentType();
            if (ct != null) outHeaders.setContentType(ct);

            // pass through successful body bytes as-is
            return new ResponseEntity<>(resp.getBody(), outHeaders, resp.getStatusCode());
        } catch (RestClientResponseException rce) {
            long t1 = System.nanoTime();
            double elapsedMs = (t1 - t0) / 1_000_000.0;
            System.err.println("[AnalysisController] RestClientResponseException after " + elapsedMs + "ms: status=" + rce.getRawStatusCode() + " body=" + rce.getResponseBodyAsString());
            rce.printStackTrace();
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(("{\"error\":\"AI Core returned " + rce.getRawStatusCode() + "\"}").getBytes(StandardCharsets.UTF_8));
        } catch (Exception ex) {
            long t1 = System.nanoTime();
            double elapsedMs = (t1 - t0) / 1_000_000.0;
            System.err.println("[AnalysisController] Exception while forwarding report after " + elapsedMs + "ms: " + ex);
            ex.printStackTrace();
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(("{\"error\":\"forward-failed\",\"message\":\"" + ex.getMessage() + "\"}").getBytes(StandardCharsets.UTF_8));
        }
    }

    // Return datasets list from AI Core
    @GetMapping("/datasets")
    public ResponseEntity<String> listDatasets() {
        String url = aiCoreUrl + "/datasets";
        try {
            return ResponseEntity.ok(rest.getForObject(url, String.class));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("{\"error\":\"AI Core unavailable\"}");
        }
    }

    // Return files for a dataset. This forwards to /files?dataset=...
    @GetMapping("/datasets/{dataset:.+}")
    public ResponseEntity<Object> listDatasetFiles(@PathVariable("dataset") String dataset) {
        System.out.println("[AnalysisController] ENTER listDatasetFiles dataset=" + dataset);
        String url = aiCoreUrl + "/files?dataset=" + URLEncoder.encode(dataset, StandardCharsets.UTF_8);
        System.out.println("[AnalysisController] Forwarding to AI Core URL: " + url);
        try {
            Object body = rest.getForObject(url, Object.class);
            System.out.println("[AnalysisController] AI Core payload type: " + (body == null ? "null" : body.getClass().getName()));
            // If AI Core returned a String that itself is JSON, try to parse to object
            if (body instanceof String) {
                String s = ((String) body).trim();
                if ((s.startsWith("{") || s.startsWith("[")) ) {
                    try {
                        ObjectMapper mapper = new ObjectMapper();
                        Object parsed = mapper.readValue(s, Object.class);
                        System.out.println("[AnalysisController] Unquoted String body -> parsed as " + (parsed==null?"null":parsed.getClass().getName()));
                        return ResponseEntity.ok(parsed);
                    } catch (Exception e) {
                        System.err.println("[AnalysisController] Failed parsing string-body: " + e.getMessage());
                    }
                }
            }
            return ResponseEntity.ok(body);
        } catch (RestClientResponseException ex) {
            System.err.println("[AnalysisController] RestClientResponseException: code=" + ex.getRawStatusCode()
                    + " body=" + ex.getResponseBodyAsString());
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(Collections.singletonMap("error", "AI Core returned error: " + ex.getRawStatusCode()));
        } catch (Exception ex) {
            ex.printStackTrace();
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(Collections.singletonMap("error", "AI Core unavailable: " + ex.getMessage()));
        }
    }

    // Forward /graph to AI Core /graph
    @GetMapping("/graph")
    public ResponseEntity<String> graph() {
        String url = aiCoreUrl + "/graph";
        try {
            return ResponseEntity.ok(rest.getForObject(url, String.class));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("{\"error\":\"AI Core unavailable\"}");
        }
    }

    // Forward consumers query to AI Core /api/v1/consumers
    @GetMapping("/consumers")
    public ResponseEntity<String> consumers(@RequestParam String service, @RequestParam(required = false) String path) {
        // Note: server expects /api/v1/consumers
        StringBuilder url = new StringBuilder(aiCoreUrl + "/api/v1/consumers?service=" + URLEncoder.encode(service, StandardCharsets.UTF_8));
        if (path != null && !path.isEmpty()) {
            url.append("&path=").append(URLEncoder.encode(path, StandardCharsets.UTF_8));
        }
        try {
            return ResponseEntity.ok(rest.getForObject(url.toString(), String.class));
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("{\"error\":\"AI Core unavailable\"}");
        }
    }

    // Forward versioning lookup to AI Core /versioning
    @GetMapping("/versioning")
    public ResponseEntity<String> versioning(@RequestParam String pairId) {
        // The server now exposes /versioning?pair_id=...
        String url = aiCoreUrl + "/versioning?pair_id=" + URLEncoder.encode(pairId, StandardCharsets.UTF_8);
        try {
            return ResponseEntity.ok(rest.getForObject(url, String.class));
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("{\"error\":\"AI Core unavailable\"}");
        }
    }

    // Forward training payload to AI Core /train
    @PostMapping("/train")
    public ResponseEntity<String> train(@RequestBody String samplesJson) {
        String url = aiCoreUrl + "/train";
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> entity = new HttpEntity<>(samplesJson, headers);
            return rest.exchange(url, HttpMethod.POST, entity, String.class);
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body("{\"error\":\"AI Core unavailable\"}");
        }
    }
 // Forward single ACE JSON to UI: /analysis/ace?pair_id=...&ace_id=...
    @GetMapping("/ace")
    public ResponseEntity<byte[]> ace(
            @RequestParam(required = false, name = "pair_id") String pairId,
            @RequestParam(name = "ace_id") String aceId) {

        StringBuilder url = new StringBuilder(aiCoreUrl + "/ace?ace_id=" + aceId);
        if (pairId != null && !pairId.isEmpty()) {
            url.append("&pair_id=").append(pairId);
        }

        try {
            ResponseEntity<byte[]> resp = rest.exchange(url.toString(), HttpMethod.GET, null, byte[].class);
            HttpHeaders outHeaders = new HttpHeaders();
            MediaType ct = resp.getHeaders().getContentType();
            if (ct != null) outHeaders.setContentType(ct);
            return new ResponseEntity<>(resp.getBody(), outHeaders, resp.getStatusCode());
        } catch (RestClientResponseException rce) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(("{\"error\":\"AI Core returned " + rce.getRawStatusCode() + "\"}").getBytes(StandardCharsets.UTF_8));
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(("{\"error\":\"AI Core unavailable\",\"message\":\"" + ex.getMessage() + "\"}").getBytes(StandardCharsets.UTF_8));
        }
    }




}
