# vLLM-Moet DSv4.1 served image + Engram tables read from the checkpoint on disk (VLLM_MOET_ENGRAM_DISK=1).
ARG BASE_IMAGE=vllm-moet-sm120:dsv41-nightly-df3bd094-r13
FROM ${BASE_IMAGE}
COPY engram_disk.c patch_vllm_engram_disk.py /opt/vllm-moet/engram_disk/
RUN gcc -O2 -fPIC -shared -pthread -o /opt/vllm-moet/engram_disk/libengram_disk.so /opt/vllm-moet/engram_disk/engram_disk.c \
 && python3 /opt/vllm-moet/engram_disk/patch_vllm_engram_disk.py "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')" \
 && python3 -c "import vllm.models.deepseek_v41.common.engram as e; assert hasattr(e, '_MoetEngramDisk'); import vllm.model_executor.model_loader.weight_utils as w; assert hasattr(w, '_moet_should_skip_weight'); print('engram disk ok')"
ENV VLLM_MOET_ENGRAM_DISK=1
LABEL com.vllm-moet.engram-disk="1"
