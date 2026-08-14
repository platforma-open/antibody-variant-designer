import { platforma } from "@platforma-open/milaboratories.antibody-variant-designer.model";
import { defineAppV3 } from "@platforma-sdk/ui-vue";
import ParentsPage from "./pages/ParentsPage.vue";
import SkippedPage from "./pages/SkippedPage.vue";
import VariantsPage from "./pages/VariantsPage.vue";

export const sdkPlugin = defineAppV3(platforma, () => ({
  routes: {
    "/": () => VariantsPage,
    "/parents": () => ParentsPage,
    "/skipped": () => SkippedPage,
  },
}));

export const useApp = sdkPlugin.useApp;
