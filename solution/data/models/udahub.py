import enum
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Enum,
    Integer,
    ForeignKey,
    UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.decl_api import DeclarativeBase
from sqlalchemy.sql import func


Base:DeclarativeBase = declarative_base()

class Account(Base):
    __tablename__ = 'accounts'
    account_id = Column(String, primary_key=True)
    account_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    users = relationship("User", back_populates="account")
    tickets = relationship("Ticket", back_populates="account")
    knowledge_articles = relationship("Knowledge", back_populates="account")

    def __repr__(self):
        return f"<Account(account_id='{self.account_id}', account_name='{self.account_name}')>"


class User(Base):
    __tablename__ = 'users'

    user_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    external_user_id = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    account = relationship("Account", back_populates="users")
    tickets = relationship("Ticket", back_populates="user")

    __table_args__ = (
        UniqueConstraint('account_id', 'external_user_id', name='uq_user_external_per_account'),
    )

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', user_name='{self.user_name}', external_user_id='{self.external_user_id}')>"



class Ticket(Base):
    __tablename__ = 'tickets'
    ticket_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    channel = Column(String)
    created_at = Column(DateTime, default=func.now())

    account = relationship("Account", back_populates="tickets")
    user = relationship("User", back_populates="tickets")
    ticket_metadata = relationship("TicketMetadata", uselist=False, back_populates="ticket")
    messages = relationship("TicketMessage", back_populates="ticket")

    def __repr__(self):
        return f"<Ticket(ticket_id='{self.ticket_id}', channel='{self.channel}', created_at='{self.created_at}')>"


class TicketMetadata(Base):
    __tablename__ = 'ticket_metadata'
    ticket_id = Column(String, ForeignKey('tickets.ticket_id'), primary_key=True)
    status = Column(String, nullable=False)
    main_issue_type = Column(String)
    tags = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    ticket = relationship("Ticket", back_populates="ticket_metadata")

    def __repr__(self):
        return f"<TicketMetadata(ticket_id='{self.ticket_id}', status='{self.status}', issue_type='{self.main_issue_type}')>"


class RoleEnum(enum.Enum):
    user = "user"
    agent = "agent"
    ai = "ai"
    system = "system"


class TicketMessage(Base):
    __tablename__ = 'ticket_messages'
    message_id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey('tickets.ticket_id'), nullable=False)
    role = Column(Enum(RoleEnum, name="role_enum"), nullable=False)
    content = Column(Text)
    created_at = Column(DateTime, default=func.now())

    ticket = relationship("Ticket", back_populates="messages")

    def __repr__(self):
        short_content = (self.content[:30] + "...") if self.content and len(self.content) > 30 else self.content
        return f"<TicketMessage(message_id='{self.message_id}', role='{self.role.name}', content='{short_content}')>"


class Knowledge(Base):
    __tablename__ = 'knowledge'
    article_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    account = relationship("Account", back_populates="knowledge_articles")

    def __repr__(self):
        return f"<Knowledge(article_id='{self.article_id}', title='{self.title}')>"


class CustomerMemory(Base):
    """Long-term memory: durable facts and preferences about a customer,
    recalled across tickets and sessions."""
    __tablename__ = 'customer_memory'
    memory_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    category = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_ticket_id = Column(String, ForeignKey('tickets.ticket_id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<CustomerMemory(user_id='{self.user_id}', category='{self.category}', content='{self.content[:30]}')>"


class RateLimitEvent(Base):
    """One counted event (a message, a verification code or LLM tokens) for the rate and cost limits.
    The key is a hash, so emails and IP addresses are not stored in plain text."""
    __tablename__ = 'rate_limit_events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=False, index=True)
    amount = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, index=True)

    def __repr__(self):
        return f"<RateLimitEvent(key='{self.key[:8]}', amount={self.amount}, created_at='{self.created_at}')>"
