{{- define "npu-lora.name" -}}
{{- printf "npu-lora-%s" (.Values.train.name | lower | replace "_" "-" | replace "." "-") | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "npu-lora.out" -}}
{{- if .Values.train.out -}}
{{- .Values.train.out -}}
{{- else -}}
{{- printf "%s/adapters/%s/%s" .Values.store.mountPath (.Values.train.model | replace "/" "--") .Values.train.name -}}
{{- end -}}
{{- end -}}

{{- define "npu-lora.labels" -}}
app: {{ include "npu-lora.name" . }}
app.kubernetes.io/name: npu-lora
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: npu-lora
npu-lora/run: {{ .Values.train.name | quote }}
{{- end -}}
