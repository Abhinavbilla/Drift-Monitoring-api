import { EmptyState } from "../components/ui";

export function ComingSoonPage({ title }: { title: string }) {
  return (
    <div>
      <EmptyState title={title} description="This page is being built in the next commit." />
    </div>
  );
}
