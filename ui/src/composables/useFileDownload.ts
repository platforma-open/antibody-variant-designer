import type { RemoteBlobHandleAndSize } from "@platforma-sdk/model";
import { getRawPlatformaInstance } from "@platforma-sdk/model";
import { ref } from "vue";

/** Pull a remote blob handle's bytes through the blob driver and trigger a
 *  browser download — a block output handle carries no URL of its own. */
export function useFileDownload() {
  const downloading = ref(false);

  async function download(handle: RemoteBlobHandleAndSize, fileName: string, mimeType: string) {
    downloading.value = true;
    try {
      const pl = getRawPlatformaInstance();
      const content = await pl.blobDriver.getContent(handle.handle);
      const blob = new Blob([new Uint8Array(content)], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } finally {
      downloading.value = false;
    }
  }

  return { downloading, download };
}
