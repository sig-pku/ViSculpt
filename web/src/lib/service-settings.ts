import type { ServiceHealth } from "@/lib/workflow-types";

export interface RuntimeServiceSettings {
  blender_rpc_url: string;
  sam3_url: string;
}

export interface ServiceSettingsResponse {
  settings: RuntimeServiceSettings;
  health?: ServiceHealth;
  persisted?: boolean;
}
