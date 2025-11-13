import base64
from typing import Any

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Attachment,
    Disposition,
    Email,
    FileContent,
    FileName,
    FileType,
    Mail,
    To,
)


class EmailClient:
    def __init__(self, api_key: str, sender: str):
        self.api_key = api_key
        self.sender = sender

    def __call__(
        self,
        to_addr: str,
        subject: str,
        body_text: str,
        filename: str,
        file_content: str,
    ) -> dict[str, Any]:
        msg = Mail(
            from_email=Email(self.sender),
            to_emails=[To(to_addr)],
            subject=subject,
            plain_text_content=body_text,
        )
        encoded = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
        attachment = Attachment(
            FileContent(encoded),
            FileName(filename),
            FileType('text/plain'),
            Disposition('attachment'),
        )
        msg.add_attachment(attachment)
        client = SendGridAPIClient(self.api_key)
        resp = client.send(msg)
        return {'status_code': resp.status_code, 'headers': dict(resp.headers)}
