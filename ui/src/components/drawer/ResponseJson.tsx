import { usePipelineStore } from '../../store/usePipelineStore';

export function ResponseJson() {
  const result = usePipelineStore((s) => s.result);

  if (!result) {
    return <div className="drawer-empty">Run the pipeline to see the response.</div>;
  }

  const json = JSON.stringify(result, null, 2);

  // Simple syntax highlighting
  const highlighted = json
    .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="json-string">"$1"</span>')
    .replace(/: (-?\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
    .replace(/: (true|false)/g, ': <span class="json-bool">$1</span>')
    .replace(/: (null)/g, ': <span class="json-null">$1</span>');

  const handleCopy = () => {
    navigator.clipboard.writeText(json);
  };

  return (
    <div className="json-viewer">
      <button className="copy-btn" onClick={handleCopy} title="Copy JSON">
        📋 Copy
      </button>
      <pre dangerouslySetInnerHTML={{ __html: highlighted }} />
    </div>
  );
}
