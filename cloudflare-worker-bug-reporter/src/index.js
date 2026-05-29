// Cloudflare Worker: TRACE bug report → GitHub Issues
// Accepts POST { description, sysinfo, trace_version, artifacts?, screenshot? }
// from the TRACE desktop app and creates an issue in alexmpdx/TRACE.
//
// `artifacts` is an optional object with any of these string-valued keys:
//   run_log, settings_yaml, manifest_json, startup_log
// Each is wrapped in a collapsed <details> block in the issue body.
//
// `screenshot` is an optional object: { content_base64, mime_type? }.
// When present, the Worker uploads the PNG to the `bug-attachments` branch
// of alexmpdx/TRACE via the Contents API, then embeds the raw URL near the
// top of the issue body. Upload failures are logged but never block the
// issue from being created — the bug report still goes through.
//
// Secrets required (set via `wrangler secret put`):
//   GITHUB_PAT — fine-grained PAT for alexmpdx/TRACE with:
//                  Issues:   Read and Write   (issue creation)
//                  Contents: Read and Write   (screenshot upload to bug-attachments branch)

const ARTIFACT_SECTIONS = [
  { key: "run_log",       label: "run.log",                     lang: "",     max: 20000 },
  { key: "settings_yaml", label: "settings.yaml",               lang: "yaml", max: 8000 },
  { key: "manifest_json", label: "_run_state.json (manifest)",  lang: "json", max: 8000 },
  { key: "startup_log",   label: "startup.log",                 lang: "",     max: 8000 },
];

// GitHub issue body cap is 65,536 chars. Leave headroom for whatever
// markdown / fence boilerplate we wrap around the user content.
const MAX_BODY_CHARS = 60000;

// Cap the base64 payload from the client. 4MB base64 ≈ 3MB binary —
// enough for any reasonable PNG of a desktop window, while preventing
// abuse via huge uploads.
const MAX_SCREENSHOT_B64_CHARS = 4 * 1024 * 1024;

const ATTACHMENTS_BRANCH = "bug-attachments";

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders() });
    }
    if (request.method !== "POST") {
      return jsonError("Method not allowed.", 405);
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return jsonError("Invalid JSON.", 400);
    }

    const description = (payload.description || "").toString().trim();
    const sysinfo = (payload.sysinfo || "").toString().trim();
    const traceVersion = (payload.trace_version || "unknown").toString().trim();
    const artifacts = (payload.artifacts && typeof payload.artifacts === "object")
      ? payload.artifacts : {};
    const screenshot = (payload.screenshot && typeof payload.screenshot === "object")
      ? payload.screenshot : null;

    if (description.length < 20) {
      return jsonError("Description must be at least 20 characters.", 400);
    }
    if (description.length > 10000) {
      return jsonError("Description must be at most 10000 characters.", 400);
    }
    if (screenshot && typeof screenshot.content_base64 === "string"
        && screenshot.content_base64.length > MAX_SCREENSHOT_B64_CHARS) {
      return jsonError("Screenshot exceeds size cap (~3 MB binary).", 400);
    }

    // Upload screenshot first so we can inline its URL in the body. If the
    // upload fails, we still create the issue without the image — the
    // failure reason goes into the body as a transparency note.
    let screenshotUrl = null;
    let screenshotError = null;
    if (screenshot && typeof screenshot.content_base64 === "string"
        && screenshot.content_base64.length > 0) {
      const uploadResult = await uploadScreenshot(env, screenshot);
      if (uploadResult.url) {
        screenshotUrl = uploadResult.url;
      } else {
        screenshotError = uploadResult.error || "unknown upload error";
        console.error(`Screenshot upload failed: ${screenshotError}`);
      }
    }

    const titleFirstLine = description.split("\n")[0].slice(0, 80);
    const title = `[bug] ${titleFirstLine}`;

    const bodyLines = [
      "**Description**",
      description,
      "",
      `**TRACE version:** ${traceVersion}`,
    ];

    if (screenshotUrl) {
      bodyLines.push("", "**Screenshot**", "", `![Screenshot](${screenshotUrl})`);
    } else if (screenshotError) {
      bodyLines.push("",
        `_(Screenshot upload failed: ${screenshotError}. The report was filed without it.)_`);
    }

    if (sysinfo) {
      bodyLines.push("", "**System info**", "```", sysinfo, "```");
    }

    for (const { key, label, lang, max } of ARTIFACT_SECTIONS) {
      let content = (artifacts[key] || "").toString();
      if (!content) continue;
      if (content.length > max) {
        content = `... [truncated to last ${max} chars]\n` + content.slice(-max);
      }
      bodyLines.push("");
      bodyLines.push(`<details><summary>${label}</summary>`);
      bodyLines.push("");
      bodyLines.push("```" + lang);
      bodyLines.push(content);
      bodyLines.push("```");
      bodyLines.push("</details>");
    }

    bodyLines.push("", "_Submitted via TRACE Report-a-Bug._");

    let body = bodyLines.join("\n");
    if (body.length > MAX_BODY_CHARS) {
      body = body.slice(0, MAX_BODY_CHARS) + "\n\n... [body truncated]";
    }

    const ghResp = await fetch(
      "https://api.github.com/repos/alexmpdx/TRACE/issues",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.GITHUB_PAT}`,
          "Accept": "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "trace-bug-reporter",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title, body }),
      }
    );

    if (!ghResp.ok) {
      const text = await ghResp.text();
      console.error(`GitHub error ${ghResp.status}: ${text.slice(0, 200)}`);
      return jsonError(`GitHub returned ${ghResp.status}.`, 502);
    }

    const issue = await ghResp.json();
    return new Response(
      JSON.stringify({
        ok: true,
        issue_url: issue.html_url,
        issue_number: issue.number,
        screenshot_url: screenshotUrl,
        screenshot_error: screenshotError,
      }),
      { status: 200, headers: jsonHeaders() }
    );
  },
};

// Upload the PNG (already base64-encoded by the client) to the
// bug-attachments branch via the Contents API. Returns either
// { url: "<raw URL>" } or { error: "<reason>" }.
async function uploadScreenshot(env, screenshot) {
  const filename = `${crypto.randomUUID()}.png`;
  const uploadResp = await fetch(
    `https://api.github.com/repos/alexmpdx/TRACE/contents/${filename}`,
    {
      method: "PUT",
      headers: {
        "Authorization": `Bearer ${env.GITHUB_PAT}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "trace-bug-reporter",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: `Upload screenshot for bug report (${filename})`,
        content: screenshot.content_base64,
        branch: ATTACHMENTS_BRANCH,
      }),
    }
  );
  if (!uploadResp.ok) {
    const text = await uploadResp.text();
    return { error: `HTTP ${uploadResp.status}: ${text.slice(0, 200)}` };
  }
  // raw.githubusercontent.com serves any committed file on a public repo.
  // Pattern: raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>
  return {
    url: `https://raw.githubusercontent.com/alexmpdx/TRACE/${ATTACHMENTS_BRANCH}/${filename}`,
  };
}

function jsonError(message, status) {
  return new Response(JSON.stringify({ ok: false, error: message }), {
    status,
    headers: jsonHeaders(),
  });
}

function jsonHeaders() {
  return { "Content-Type": "application/json", ...corsHeaders() };
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}
